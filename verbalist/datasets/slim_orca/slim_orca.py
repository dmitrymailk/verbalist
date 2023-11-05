


from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy import Column, Integer, String
from tqdm import tqdm
import json
from datasets import load_dataset, Dataset
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine
from sqlalchemy.pool import SingletonThreadPool
from sqlalchemy import text

import torch
dataset = load_dataset("dim/SlimOrcaEN")
dataset = dataset["train"]
dataset = dataset.to_list()


engine = create_engine(
    open("./postgre_creds").read()
)

# with engine.begin() as conn:
#     conn.execute(text('PRAGMA journal_mode=WAL'))


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/wmt21-dense-24-wide-en-x",
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_4bit=True
    # use_flash_attention_2=True
)
model.eval()
# model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/wmt21-dense-24-wide-en-x",
)

inputs = tokenizer(
    "wmtdata newsdomain One model for many languages.", return_tensors="pt"
)

generated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.get_lang_id("ru"),
)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


class Base(DeclarativeBase):
    pass


class SlimOrcaTranslation(Base):
    __tablename__ = "slim_orca"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(Integer, unique=True)
    json = Column(String)


# создаем таблицы
Base.metadata.create_all(bind=engine)
with torch.no_grad():
    # создаем сессию подключения к бд
    with Session(autoflush=False, bind=engine) as db:
        # for key in tqdm(range(3_00_000, len(dataset))):
        for key in tqdm(range(1_00_000, len(dataset))):
        # for key in tqdm(range(28000, len(dataset))):
            trans = (
                db.query(SlimOrcaTranslation).filter(SlimOrcaTranslation.key == key).first()
            )
            # print(trans)
            if trans is None:
                try:
                    item = dataset[key]
                    print(f"KEY={key}")
                    for i in range(len(item["conversations"])):
                        value_en = item["conversations"][i]["value"].split("\n")
                        value_en = [item for item in value_en if len(item) > 0]
                        # print(item['conversations'][i]['value'])
                        value_ru = tokenizer.batch_encode_plus(
                            value_en,
                            return_tensors="pt",
                            padding=True,
                        ).to("cuda")

                        value_ru = model.generate(
                            **value_ru,
                            forced_bos_token_id=tokenizer.get_lang_id("ru"),
                            # num_beams=2
                        )
                        value_ru = tokenizer.batch_decode(
                            value_ru,
                            skip_special_tokens=True,
                        )
                        print(f"VALUE_EN={value_en}")
                        value_ru = "\n".join(value_ru)
                        item["conversations"][i]["value_ru"] = value_ru
                        print(f"VALUE_RU={value_ru}")
                        print("-" * 10)
                        print("-" * 10)
                    # break
                    # создаем объект Person для добавления в бд
                    row = SlimOrcaTranslation(
                        key=key,
                        json=json.dumps(
                            item,
                            ensure_ascii=False,
                        ),
                    )
                    db.add(row)  # добавляем в бд
                    db.commit()
                except:
                    row = SlimOrcaTranslation(
                        key=key,
                        json="",
                    )
                    db.add(row)  # добавляем в бд
                    db.commit()
                    

    # dataset[0]