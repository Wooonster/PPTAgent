import json

import PIL.Image
from rich import print

import llms
from presentation import Picture, Presentation
from utils import Config, pbasename, pexists, pjoin


class ImageLabler:
    def __init__(self, presentation: Presentation, config: Config):
        self.presentation = presentation
        self.slide_area = presentation.slide_width.pt * presentation.slide_height.pt
        self.image_stats = {}
        self.stats_file = pjoin(config.RUN_DIR, "image_stats.json")
        self.config = config
        self.collect_images()
        if pexists(self.stats_file):
            image_stats: dict[str, dict] = json.load(open(self.stats_file, "r"))
            for name, stat in image_stats.items():
                if pbasename(name) in self.image_stats:
                    self.image_stats[pbasename(name)] = stat

    def apply_stats(self):
        '''将生成的描述应用到幻灯片中的图片形状'''
        for slide in self.presentation.slides:
            for shape in slide.shape_filter(Picture):
                stats = self.image_stats[pbasename(shape.img_path)]
                shape.caption = stats["caption"]

    def caption_images(self):
        '''用 vlm 为没有描述的图片生成描述'''
        caption_prompt = open("../prompts/caption.txt").read()
        for image, stats in self.image_stats.items():
            if "caption" not in stats:
                stats["caption"] = llms.vision_model(
                    caption_prompt, pjoin(self.config.IMAGE_DIR, image)
                )
                print("captioned", image, ": ", stats["caption"])
        json.dump(
            self.image_stats,
            open(self.stats_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
        self.apply_stats()
        return self.image_stats

    def collect_images(self):
        '''从演示文稿中提取图片信息并生成统计数据
        
        统计信息包括：图片出现的次数、所在的幻灯片编号集合、相对于幻灯片总面积的百分比、图片的尺寸（宽度和高度）
        '''
        for slide_index, slide in enumerate(self.presentation.slides):
            for shape in slide.shape_filter(Picture):
                image_path = pbasename(shape.data[0])
                self.image_stats[image_path] = {
                    "appear_times": 0,
                    "slide_numbers": set(),
                    "relative_area": shape.area / self.slide_area * 100,
                    "size": PIL.Image.open(
                        pjoin(self.config.IMAGE_DIR, image_path)
                    ).size,
                }
                self.image_stats[image_path]["appear_times"] += 1
                self.image_stats[image_path]["slide_numbers"].add(slide_index + 1)

        for image_path, stats in self.image_stats.items():
            stats["slide_numbers"] = sorted(list(stats["slide_numbers"]))
            ranges = self._find_ranges(stats["slide_numbers"])
            # 生成图片在幻灯片中的连续范围
            top_ranges = sorted(ranges, key=lambda x: x[1] - x[0], reverse=True)[:3]
            top_ranges_str = ", ".join(
                [f"{r[0]}-{r[1]}" if r[0] != r[1] else f"{r[0]}" for r in top_ranges]
            )
            stats["top_ranges_str"] = top_ranges_str

    def _find_ranges(self, numbers):
        '''根据图片的幻灯片编号，生成连续范围'''
        ranges = []
        start = numbers[0]
        end = numbers[0]
        for num in numbers[1:]:
            if num == end + 1:
                end = num
            else:
                ranges.append((start, end))
                start = num
                end = num
        ranges.append((start, end))
        return ranges
