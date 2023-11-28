import argparse
import glob
import json
import os
import shutil
import datasets
import numpy as np


_TASKS = {}


def get_files_from_dir(dir_path):
    f = []
    for dir_path, dirn_ames, filenames in os.walk(dir_path):
        for fn in filenames:
            fn = os.path.join(dir_path, fn)
            f.append(fn)
    return f


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        text = json.loads(file.read().strip())
    return text


def register_task(cls):
    _TASKS[cls.__name__] = cls
    return cls


class BaseTask(object):
    @property
    def src_name(self):
        return self.__class__.__name__.lower()

    @property
    def dst_name(self):
        return self.__class__.__name__

    @property
    def outputs_path(self):
        return os.path.join(self.outputs_dir, f"lm_harness_logs_{self.src_name}", "output_answers.json")

    @property
    def submission_path(self):
        return os.path.join(self.dst_dir, f"{self.dst_name}.json")

    @staticmethod
    def doc_to_meta(doc):
        return doc["meta"]

    def doc_to_id(self, doc):
        return self.doc_to_meta(doc)["id"]

    def load(self):
        dataset = datasets.load_dataset(path="ai-forever/MERA", name=self.src_name)["test"]
        examples = dict()
        for example in dataset:
            doct_id = self.doc_to_id(example)
            examples[doct_id] = example
        return examples

    def __init__(self, outputs_dir, dst_dir):
        self.outputs_dir = outputs_dir
        self.dst_dir = dst_dir
        self.dataset = self.load()


class ClassificationTask(BaseTask):
    @property
    def choices(self):
        return ["0", "1"]

    def convert(self):
        submission = self.outputs_to_submission(load_json(self.outputs_path))
        save_json(submission, self.submission_path)
        return submission

    def outputs_to_submission(self, outputs):
        res = []
        for idx, (doc_id, resp) in enumerate(outputs.items()):
            doc_id = int(doc_id)
            res.append(self.doc_outputs_to_submission(doc_id, resp))
        return {"data": {"test": res}}

    @staticmethod
    def parse_doc(doc):
        return doc

    def doc_outputs_to_submission(self, doc_id, outputs):
        log_probs = np.zeros(len(outputs))
        for doc in outputs:
            idx, prob = self.parse_doc(doc)
            log_probs[idx] = prob
        idx = log_probs.argmax()
        res = {
            "outputs": self.choices[idx],
            "meta": {"id": doc_id},
        }
        return res


class TextTask(ClassificationTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        res = {
            "outputs": outputs[0][1].strip(),
            "meta": {"id": doc_id},
        }
        return res


@register_task
class BPS(ClassificationTask):
    pass


@register_task
class LCS(ClassificationTask):
    @property
    def choices(self):
        return list(map(str, range(10)))

    @staticmethod
    def parse_doc(doc):
        return doc[0], doc[1][0]


@register_task
class CheGeKa(TextTask):
    pass


@register_task
class MathLogicQA(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class MultiQ(TextTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        origin_doc = self.dataset[doc_id]
        text = outputs[0][1].strip()
        pos = origin_doc["inputs"]["support_text"].find(text)
        if -1 == pos:
            pos = origin_doc["inputs"]["text"].find(text)
        res = {
            "outputs": [
                {
                    "label": origin_doc["outputs"][0]["label"],
                    "length": len(text),
                    "offset": pos,
                    "segment": text,
                },
            ],
            "meta": {
                "id": doc_id,
            },
        }
        return res


@register_task
class PARus(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2"]


@register_task
class RCB(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2", "3"]


@register_task
class ruDetox(TextTask):
    pass


@register_task
class ruEthics(ClassificationTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        doc = super().doc_outputs_to_submission(doc_id, outputs)
        out = str(doc["outputs"])
        doc["outputs"] = {
            "virtue": out,
            "law": out,
            "moral": out,
            "justice": out,
            "utilitarianism": out,
        }
        return doc


@register_task
class ruHateSpeech(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2"]


@register_task
class ruHHH(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2"]


@register_task
class ruMMLU(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class ruModAr(TextTask):
    pass


@register_task
class ruMultiAr(TextTask):
    pass


@register_task
class SimpleAr(TextTask):
    pass


@register_task
class ruOpenBookQA(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class ruWorldTree(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class RWSD(ClassificationTask):
    @property
    def choices(self):
        return ["Да", "Нет"]


@register_task
class USE(TextTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        origin_doc = self.dataset[doc_id]
        res = {
            "outputs": outputs[0][1].strip(),
            "meta": {
                "id": doc_id,
                "id_task": origin_doc["meta"]["id_task"],
                "variant": origin_doc["meta"]["variant"],
            },
        }
        return res


@register_task
class ruTiE(TextTask):
    def load(self):
        dataset = datasets.load_dataset(path="ai-forever/MERA", name=self.src_name)["test"]
        dataset = [list(dataset)]
        return dataset

    @property
    def choices(self):
        return ["1", "2"]

    def outputs_to_submission(self, outputs):
        res_by_qid = dict()
        for idx, (question_id, resp) in enumerate(outputs.items()):
            question_id = int(question_id)
            res_by_qid[question_id] = self.doc_outputs_to_submission(question_id, resp)
        res = []
        for dialog in self.dataset:
            new_dialog = []
            for question in dialog:
                question_id = question["meta"]["question_id"]
                new_question = {
                    "outputs": res_by_qid[question_id],
                    "meta": {
                        "dialog_id": question["meta"]["dialog_id"],
                        "question_id": question_id,
                    },
                }
                new_dialog.append(new_question)
            res.append(new_dialog)

        return {"data": {"test": res}}

    def doc_outputs_to_submission(self, doc_id, outputs):
        log_probs = np.zeros(len(outputs))
        for doc in outputs:
            idx, prob = self.parse_doc(doc)
            log_probs[idx] = prob
        idx = log_probs.argmax()
        return self.choices[idx]


@register_task
class ruHumanEval(TextTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        res = {
            "outputs": outputs,
            "meta": {
                "id": doc_id,
            },
        }
        return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, help="lm harness outputs")
    parser.add_argument("--dst_dir", type=str, default="submission/", help="dir to save files for submission")
    parser.add_argument(
        "--logs_public_submit",
        type=bool,
        default=True,
        help="pack logs for public submission in separate file",
        action=argparse.BooleanOptionalAction,
    )
    res = parser.parse_known_args()[0]
    return res


def create_submission(outputs_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    no_tasks = []
    for task_name, task_cls in _TASKS.items():
        task = task_cls(outputs_dir=outputs_dir, dst_dir=dst_dir)
        _ = task.convert()
        print("---------------------")
    print("Not refactored tasks", no_tasks)
    zip_path = shutil.make_archive(dst_dir, "zip", dst_dir)
    print("Submission stored at", zip_path)
    return no_tasks


def pack_submission_logs(outputs_dir: str, dst_dir: str):
    if os.path.isdir(outputs_dir):
        zip_dir = f"{dst_dir}_logs_public"
        os.makedirs(zip_dir, exist_ok=True)
        for file_path in glob.glob(os.path.join(outputs_dir, "*.json")) + glob.glob(
            os.path.join(outputs_dir, "rutie/*.json")
        ):
            shutil.copy2(file_path, zip_dir)
        zip_path = shutil.make_archive(zip_dir, "zip", zip_dir)
        shutil.rmtree(zip_dir)
        print("Logs to add with public submission stored at", zip_path)
    else:
        raise ValueError(f"{outputs_dir} is not directory")


def main():
    args = get_args()
    _ = create_submission(args.outputs_dir, args.dst_dir)
    if args.logs_public_submit:
        print("Packing logs for public submission...")
        pack_submission_logs(args.outputs_dir, args.dst_dir)


if __name__ == "__main__":
    main()
