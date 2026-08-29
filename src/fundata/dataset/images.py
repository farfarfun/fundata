import json

import pandas as pd

from fundata._util import path_parse

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("max_colwidth", 500)


def dataset_coco(data_root="./download/coco/"):
    # step1 (download+decompress raw COCO archives) relied on download+decompress
    # helpers that no longer exist and was already disabled (never called)
    # before this fix; removed rather than reimplemented.
    def step2(json_path, target_file):
        json_data = json.load(open(json_path))
        df_annotations = pd.DataFrame.from_dict(json_data["annotations"])

        df_categories = pd.DataFrame.from_dict(json_data["categories"])
        df_categories.reset_index(inplace=True)

        df_annotations = pd.merge(
            df_annotations,
            df_categories[["id", "index"]],
            left_on="category_id",
            right_on="id",
            suffixes=("", "_y"),
        ).drop(["id_y"], axis=1)

        df_annotations["xmin"] = df_annotations["bbox"].apply(lambda x: int(x[0]))
        df_annotations["ymin"] = df_annotations["bbox"].apply(lambda x: int(x[1]))
        df_annotations["xmax"] = df_annotations["bbox"].apply(
            lambda x: int(x[2] + x[0])
        )
        df_annotations["ymax"] = df_annotations["bbox"].apply(
            lambda x: int(x[3] + x[1])
        )

        df_annotations["label"] = df_annotations["bbox"].apply(
            lambda x: "{},{},{},{},".format(
                int(x[0]), int(x[1]), int(x[2] + x[0]), int(x[3] + x[1])
            )
        ) + df_annotations["index"].astype("str")
        df_annotations["image_path"] = df_annotations["image_id"].apply(
            lambda x: data_root + "/images/{}.jpg".format(x)
        )

        df_annotations["image_path"] = df_annotations["image_id"].apply(
            lambda x: data_root + "/images/{}{}.jpg".format("0" * (12 - len(str(x))), x)
        )

        df_res = pd.DataFrame(
            df_annotations[["image_path", "label"]]
            .groupby(["image_path"])["label"]
            .apply(list)
            .apply(lambda x: " ".join(x))
        ).reset_index()

        df_res.to_csv(
            target_file,
            header=None,
            index=None,
            sep=" ",
            quotechar=" ",
        )

    # step1()
    data_root = path_parse(data_root)
    step2(
        json_path=data_root + "/annotations/instances_train2017.json",
        target_file=data_root + "train2017.txt",
    )
    step2(
        json_path=data_root + "/annotations/instances_val2017.json",
        target_file=data_root + "val2017.txt",
    )
