import json
import os
import shutil
from collections import defaultdict

from jinja2 import Template

import llms
from model_utils import get_cluster, get_image_embedding, images_cosine_similarity
from presentation import Presentation
from utils import Config, pexists, pjoin, tenacity

import logging
from fastapi.logger import logger

# 设置日志级别为 INFO
logging.basicConfig(level=logging.INFO)


class SlideInducter:
    def __init__(
        self,
        prs: Presentation,
        ppt_image_folder: str,
        template_image_folder: str,
        config: Config,
        image_models: list,
    ):
        self.prs = prs
        self.config = config
        self.ppt_image_folder = ppt_image_folder
        self.template_image_folder = template_image_folder
        assert (
            len(os.listdir(template_image_folder))
            == len(prs)
            == len(os.listdir(ppt_image_folder))
        )
        self.image_models = image_models
        self.slide_induction = defaultdict(lambda: defaultdict(list))
        model_identifier = llms.get_simple_modelname(
            [llms.language_model, llms.vision_model]
        )
        self.output_dir = pjoin(config.RUN_DIR, "template_induct", model_identifier)
        self.split_cache = pjoin(self.output_dir, f"split_cache.json")
        self.induct_cache = pjoin(self.output_dir, f"induct_cache.json")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def layout_induct(self):
        '''对幻灯片进行布局归纳，生成幻灯片的分类结构和模板'''
        if pexists(self.induct_cache):
            return json.load(open(self.induct_cache))
        content_slides_index, functional_cluster = self.category_split()
        logger.info(f'functional_cluster.items() in induct.layout_induct:\n{functional_cluster.items()}')
        for layout_name, cluster in functional_cluster.items():
            for slide_idx in cluster:
                slide_idx = int(slide_idx[6])
                content_type = self.prs.slides[slide_idx].get_content_type()
                # content_type = self.prs.slides[slide_idx - 1].get_content_type()
                self.slide_induction[layout_name + ":" + content_type]["slides"].append(
                    slide_idx
                )
        for layout_name, cluster in self.slide_induction.items():
            cluster["template_id"] = cluster["slides"][-1]

        functional_keys = list(self.slide_induction.keys())
        function_slides_index = set()
        for layout_name, cluster in self.slide_induction.items():
            function_slides_index.update(cluster["slides"])
        used_slides_index = function_slides_index.union(content_slides_index)
        for i in range(len(self.prs.slides)):
            if i + 1 not in used_slides_index:
                content_slides_index.add(i + 1)
        self.layout_split(content_slides_index)
        if self.config.DEBUG:
            for layout_name, cluster in self.slide_induction.items():
                cluster_dir = pjoin(self.output_dir, "cluster_slides", layout_name)
                os.makedirs(cluster_dir, exist_ok=True)
                for slide_idx in cluster["slides"]:
                    shutil.copy(
                        pjoin(self.ppt_image_folder, f"slide_{slide_idx:04d}.jpg"),
                        pjoin(cluster_dir, f"slide_{slide_idx:04d}.jpg"),
                    )
        self.slide_induction["functional_keys"] = functional_keys
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slide_induction

    def category_split(self):
        '''使用 语言模型 + prompt 将幻灯片分为功能性和内容性两类, 并对ppt页面归类'''
        if pexists(self.split_cache):
            split = json.load(open(self.split_cache))
            return set(split["content_slides_index"]), split["functional_cluster"]
        category_split_template = Template(open("../prompts/category_split.txt").read())
        functional_cluster = llms.language_model(
            category_split_template.render(slides=self.prs.to_text()),
            return_json=True,
        )
        functional_slides = set(sum(functional_cluster.values(), []))
        content_slides_index = set(range(1, len(self.prs) + 1)) - functional_slides

        json.dump(
            {
                "content_slides_index": list(content_slides_index),
                "functional_cluster": functional_cluster,
            },
            open(self.split_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return content_slides_index, functional_cluster

    def layout_split(self, content_slides_index: set[int]):
        '''进一步对内容性幻灯片进行布局归纳，基于布局和内容类型进行聚类
        
        按布局名称和内容类型, 将内容性幻灯片划分为多个子类;
        使用图像余弦相似度对每个子类进行聚类
        '''
        embeddings = get_image_embedding(self.template_image_folder, *self.image_models)
        assert len(embeddings) == len(self.prs)
        template = Template(open("../prompts/ask_category.txt").read())
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:
            slide = self.prs.slides[slide_idx - 1]
            content_type = slide.get_content_type()
            layout_name = slide.slide_layout_name
            content_split[(layout_name, content_type)].append(slide_idx)

        for (layout_name, content_type), slides in content_split.items():
            sub_embeddings = [
                embeddings[f"slide_{slide_idx:04d}.jpg"] for slide_idx in slides
            ]
            similarity = images_cosine_similarity(sub_embeddings)
            for cluster in get_cluster(similarity):
                slide_indexs = [slides[i] for i in cluster]
                template_id = max(
                    slide_indexs,
                    key=lambda x: len(self.prs.slides[x - 1].shapes),
                )
                cluster_name = (
                    llms.vision_model(
                        template.render(
                            existed_layoutnames=list(self.slide_induction.keys()),
                        ),
                        pjoin(self.ppt_image_folder, f"slide_{template_id:04d}.jpg"),
                    )
                    + ":"
                    + content_type
                )
                self.slide_induction[cluster_name]["template_id"] = template_id
                self.slide_induction[cluster_name]["slides"] = slide_indexs

    @tenacity
    def content_induct(self):
        '''根据模板幻灯片提取内容模式
        
        遍历每类幻灯片的模板, 使用语言模型提取内容模式
        '''
        self.slide_induction = self.layout_induct()
        content_induct_prompt = Template(open("../prompts/content_induct.txt").read())
        for layout_name, cluster in self.slide_induction.items():
            if "template_id" in cluster and "content_schema" not in cluster:
                schema = llms.language_model(
                    content_induct_prompt.render(
                        slide=self.prs.slides[cluster["template_id"] - 1].to_html(
                            element_id=False, paragraph_id=False
                        )
                    ),
                    return_json=True,
                )

                schema = {
                    "slide_1": {
                        "name": "main title",
                        "description": "the title of the slide",
                        "type": [
                            "text"
                        ],
                        "data": [
                            "Welcome to Our Presentation"
                        ]
                    },
                    "slide_2": {
                        "name": "content bullets",
                        "description": "contain text on specific slides to provide further information or details, typically organized in bullet points",
                        "type": [
                            "text"
                        ],
                        "data": [
                            "<p>This is the first bullet point of this slide. It describes the content and objectives of what we will discuss.</p>",
                            "<p>The audience can also refer back to these points while listening to our presentation.</p>"
                        ]
                    },
                    "slide_3": {
                        "name": "acknowledgments",
                        "description": "a short statement of gratitude or thanks to those who helped with the material, such as the speakers, sponsors, etc.",
                        "type": [
                            "text"
                        ],
                        "data": [
                            "<p>Thank you for joining us today!</p>"
                        ]
                    }
                }

                logger.info(f'schema in induce.content_induct:\n{schema}\nand the schema.keys() is:\n{schema.keys()}')
                
                # 处理 schema 中可能的空字段或缺失字段
                keys_to_remove = []
                for k in list(schema.keys()):
                    if "data" not in schema[k]:
                        logger.warning(f"Missing `data` in {k}: {schema[k]}")
                        schema[k]["data"] = ["Default Value"]  # 或者设置一个默认值
                    if len(schema[k]["data"]) == 0:
                        logger.warning(f"Empty content schema for {k}: {schema[k]}")
                        keys_to_remove.append(k)
                
                # 移除不符合要求的键
                for k in keys_to_remove:
                    schema.pop(k)
                
                assert len(schema) > 0, "No content schema generated"
                self.slide_induction[layout_name]["content_schema"] = schema
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slide_induction
