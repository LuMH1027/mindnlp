import unittest
class Mask2FormerImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        size=None,
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        num_labels=10,
        do_reduce_labels=True,
        ignore_index=255,
    ):
        super().__init__()

        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = {"shortest_edge": 32, "longest_edge": 1333} if size is None else size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisor = 0
        # for the post_process_functions
        self.batch_size = 2
        self.num_queries = 3
        self.num_classes = 2
        self.height = 3
        self.width = 4
        self.num_labels = num_labels
        self.do_reduce_labels = do_reduce_labels
        self.ignore_index = ignore_index

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size_divisor": self.size_divisor,
            "num_labels": self.num_labels,
            "do_reduce_labels": self.do_reduce_labels,
            "ignore_index": self.ignore_index,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to Mask2FormerImageProcessor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[1], image.shape[2]
            if w < h:
                expected_height = int(self.size["shortest_edge"] * h / w)
                expected_width = self.size["shortest_edge"]
            elif w > h:
                expected_height = self.size["shortest_edge"]
                expected_width = int(self.size["shortest_edge"] * w / h)
            else:
                expected_height = self.size["shortest_edge"]
                expected_width = self.size["shortest_edge"]

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

    def get_fake_mask2former_outputs(self):
        return Mask2FormerForUniversalSegmentationOutput(
            # +1 for null class
            class_queries_logits=ops.randn((self.batch_size, self.num_queries, self.num_classes + 1)),
            masks_queries_logits=ops.randn((self.batch_size, self.num_queries, self.height, self.width)),
        )

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values(images, batched=True)
        return self.num_channels, height, width

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
from mindnlp.transformers import Mask2FormerImageProcessor

image_processing_class = Mask2FormerImageProcessor
image_processor_tester = Mask2FormerImageProcessingTester(self)
fature_extractor = image_processing_class(num_labels=self.image_processor_tester.num_classes)
        outputs = image_processor_tester.get_fake_mask2former_outputs()

        segmentation = fature_extractor.post_process_semantic_segmentation(outputs)