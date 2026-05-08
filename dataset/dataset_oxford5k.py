import os
import pickle
import datasets

_VERSION = datasets.Version("1.0.0")

_URLS = {
    "roxford5k": {
        "images": [
            "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"
        ],
        "ground_truth": [
            "http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl"
        ],
    },
    "rparis6k": {
        "images": [
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz",
        ],
        "ground_truth": [
            "http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl"
        ],
    },
    "revisitop1m": {
        "images": [
            f"http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.{i+1}.tar.gz"
            for i in range(100)
        ]
    },
}

_DESCRIPTION = (
    "Oxford5k, Paris6k, and RevisitOP1M benchmark datasets for image retrieval."
)

_CITATION = """\
@inproceedings{Radenovic2018RevisitingOP,
  title={Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking},
  author={Filip Radenovic and Ahmet Iscen and Giorgos Tolias and Yannis Avrithis and Ondrej Chum},
  year={2018}
}
"""

BUILDER_CONFIGS = [
    datasets.BuilderConfig(
        name="roxford5k",
        version=_VERSION,
        description="Oxford 5k image retrieval dataset.",
    ),
    datasets.BuilderConfig(
        name="rparis6k",
        version=_VERSION,
        description="Paris 6k image retrieval dataset.",
    ),
    datasets.BuilderConfig(
        name="revisitop1m",
        version=_VERSION,
        description="RevisitOP 1M distractor images.",
    ),
    datasets.BuilderConfig(
        name="oxfordparis",
        version=_VERSION,
        description="Oxford + Paris combined dataset.",
    ),
]
class RevisitOP(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "roxford5k"
    
    # 定义本地基础路径
    _LOCAL_DATA_DIR = "/mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/Oxford5k"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "filename": datasets.Value("string"),
                    "dataset": datasets.Value("string"),
                    "query_id": datasets.Value("int32"),
                    "bbx": datasets.Sequence(datasets.Value("float32")),
                    "easy": datasets.Sequence(datasets.Value("int32")),
                    "hard": datasets.Sequence(datasets.Value("int32")),
                    "junk": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=None,
            homepage="http://cmp.felk.cvut.cz/revisitop/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # 1. 指定本地图片目录和标注文件路径
        # 根据你的描述，所有 jpg 和 pkl 都在该目录下
        image_dir = self._LOCAL_DATA_DIR
        
        # 假设 Oxford5k 的标注文件名是 gnd_roxford5k.pkl
        # 如果文件名不同，请根据实际情况修改
        gt_filename = "gnd_roxford5k.pkl" 
        gt_path = os.path.join(self._LOCAL_DATA_DIR, gt_filename)

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"标注文件未找到: {gt_path}。请确认文件名是否正确。")

        # 2. 返回 SplitGenerator，直接使用本地路径作为 gen_kwargs
        return [
            datasets.SplitGenerator(
                name="qimlist",
                gen_kwargs={
                    "image_dirs": [image_dir],
                    "ground_truth_files": [gt_path],
                    "split_type": "qimlist",
                    "dataset_name": self.config.name,
                },
            ),
            datasets.SplitGenerator(
                name="imlist",
                gen_kwargs={
                    "image_dirs": [image_dir],
                    "ground_truth_files": [gt_path],
                    "split_type": "imlist",
                    "dataset_name": self.config.name,
                },
            ),
        ]

    def _generate_examples(self, image_dirs, ground_truth_files, split_type, dataset_name):
        # 该部分逻辑基本保持不变，因为它已经支持从传入的路径中通过 os.walk 搜索文件 [citation:1][citation:5]
        image_path_mapping = {}
        for image_dir in image_dirs:
            # 遍历本地目录下的所有图片文件 [citation:1][citation:2]
            for root, _, files in os.walk(image_dir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        fpath = os.path.join(root, fname)
                        fname_no_ext = os.path.splitext(fname)[0]
                        image_path_mapping[fname_no_ext] = fpath

        # 加载本地标注文件
        ground_truth_data = []
        for gt_file in ground_truth_files:
            with open(gt_file, "rb") as f:
                gt_data = pickle.load(f)
                ground_truth_data.append(gt_data)

        key = 0
        for gt_data in ground_truth_data:
            imlist = gt_data["imlist"]
            qimlist = gt_data["qimlist"]
            gnd = gt_data["gnd"]

            if split_type == "qimlist":
                for i, query_name in enumerate(qimlist):
                    query_name_no_ext = os.path.splitext(query_name)[0]
                    if query_name_no_ext in image_path_mapping:
                        query_gnd = gnd[i]
                        yield key, {
                            "image": image_path_mapping[query_name_no_ext],
                            "filename": query_name,
                            "dataset": dataset_name,
                            "query_id": i,
                            "bbx": query_gnd.get("bbx", []),
                            "easy": query_gnd.get("easy", []),
                            "hard": query_gnd.get("hard", []),
                            "junk": query_gnd.get("junk", []),
                        }
                        key += 1
            elif split_type == "imlist":
                for i, image_name in enumerate(imlist):
                    image_name_no_ext = os.path.splitext(image_name)[0]
                    if image_name_no_ext in image_path_mapping:
                        yield key, {
                            "image": image_path_mapping[image_name_no_ext],
                            "filename": image_name,
                            "dataset": dataset_name,
                            "query_id": -1,
                            "bbx": [],
                            "easy": [],
                            "hard": [],
                            "junk": [],
                        }
                        key += 1


if __name__=="__main__":
    from datasets import load_dataset
    query_dataset = load_dataset(
        "dataset/dataset_oxford5k.py", 
        name='roxford5k',
        split="qimlist",
        trust_remote_code=True,
    )

    # Load cand images
    db_dataset = load_dataset(
        "dataset/dataset_oxford5k.py", 
        name='roxford5k',
        split="imlist",
        trust_remote_code=True,
    )


    # Example query image
    query_example = query_dataset[0]
    breakpoint()