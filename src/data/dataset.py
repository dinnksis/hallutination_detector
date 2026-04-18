import json
import pandas as pd
import random
import re
from tqdm import tqdm


def load_raw_data(train_path, val_path):

    data_train = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                data_train.append({
                    "evidence": ex["evidence"],
                    "claim": ex["claim"],
                    "label": ex["label"]
                })
    df_train = pd.DataFrame(data_train)

    data_val = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                data_val.append({
                    "evidence": ex["evidence"],
                    "claim": ex["claim"],
                    "label": ex["label"]
                })
    df_val = pd.DataFrame(data_val)

    df = pd.concat([df_train, df_val], ignore_index=True)

    df = df.rename(columns={
        "evidence": "prompt",
        "claim": "model_answer",
        "label": "is_hallucination"
    })
    df["correct_answer"] = None
    df["comment"] = None

    return df

def filter_true_samples(df):
    df_true = df[df["is_hallucination"] == 0].copy().reset_index(drop=True)
    return df_true


class FactualMutator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        
        self.units = {
          
            "руб.": "долл.", "долл.": "евро", "евро": "юань", "юань": "руб.",
            "рублей": "долларов", "долларов": "евро", "евро": "фунтов",
           
            "млн": "тыс.", "тыс.": "млн", "млрд": "млн", "млн": "млрд",
            "процент": "пункт", "пункт": "процент", "%": "промилле",
    
            "км": "м", "м": "км", "см": "мм", "мм": "см",
            "миль": "км", "км": "миль",
           
            "кг": "т", "т": "кг", "г": "кг", "кг": "г",
            "литр": "мл", "мл": "литр", "куб. м": "куб. км",
           
            "лет": "месяцев", "месяцев": "недель", "дней": "часов"
        }
        
        self.negations = {
          
            "не ": "",
            "не был": "был", "не была": "была", "не было": "было", "не были": "были",
            "отсутствует": "присутствует", "присутствует": "отсутствует",
            "нет ": "есть ", "есть ": "нет ",
            "ложно": "верно", "верно": "ложно",
            "всегда": "никогда", "никогда": "всегда",
            "часто": "редко", "редко": "часто",
            "запрещено": "разрешено", "разрешено": "запрещено"
        }
        
        self.names = {
            
            "Путин": "Медведев", "Медведев": "Путин",
            "Сталин": "Ленин", "Ленин": "Сталин", "Троцкий": "Сталин",
            "Петр I": "Николай II", "Николай II": "Петр I", "Екатерина II": "Елизавета",
            "Хрущев": "Брежнев", "Брежнев": "Андропов", "Горбачев": "Ельцин", "Ельцин": "Путин",
           
            "Байден": "Трамп", "Трамп": "Байден", "Обама": "Буш",
            "Макрон": "Шольц", "Шольц": "Макрон", "Джонсон": "Сунак",
            "Си Цзиньпин": "Ли Кэцян", "Ким Чен Ын": "Ким Чен Ир",
       
            "Пушкин": "Лермонтов", "Лермонтов": "Гоголь", "Толстой": "Достоевский", 
            "Достоевский": "Тургенев", "Чехов": "Бунин",
            "Есенин": "Маяковский", "Маяковский": "Блок",
            "Бетховен": "Моцарт", "Моцарт": "Бах", "Шекспир": "Диккенс"
        }
        
        self.geo =  {
           
            "Россия": "СССР", "СССР": "Россия", "РСФСР": "СССР",
            "США": "Канада", "Канада": "Мексика", "Мексика": "Бразилия",
            "Германия": "Франция", "Франция": "Италия", "Италия": "Испания", "Испания": "Португалия",
            "Великобритания": "Англия", "Англия": "Шотландия", "Шотландия": "Уэльс",
            "Китай": "Япония", "Япония": "Корея", "Корея": "Вьетнам",
            "Украина": "Беларусь", "Беларусь": "Казахстан", "Казахстан": "Узбекистан",
            
            "Москва": "Санкт-Петербург", "Санкт-Петербург": "Новосибирск",
            "Лондон": "Париж", "Париж": "Берлин", "Берлин": "Рим", "Рим": "Мадрид",
            "Нью-Йорк": "Лос-Анджелес", "Лос-Анджелес": "Чикаго",
            "Пекин": "Шанхай", "Шанхай": "Токио", "Токио": "Сеул",
            "Киев": "Минск", "Минск": "Астана", "Астана": "Ташкент"
        }
        
        self.dates = {
            "января": "февраля", "февраля": "марта", "марта": "апреля",
            "понедельник": "вторник", "вторник": "среду", "среду": "четверг",
            "утром": "вечером", "вечером": "ночью", "ночью": "утром",
            "весной": "летом", "летом": "осенью", "осенью": "зимой"
        }


        self.quantifiers = {
            "один": "два", "два": "три", "три": "пять", "пять": "десять",
            "первый": "второй", "второй": "третий", "третий": "четвертый",
            "половина": "треть", "треть": "четверть", "четверть": "половина",
            "несколько": "множество", "множество": "большинство", "большинство": "меньшинство"
        }

    def _mutate_year(self, text):
        def change_year(m):
            return str(int(m.group(0)) + self.rng.choice([-7, -3, 2, 5, 12]))
        return re.sub(r'\b(1[0-9]{3}|20[0-2][0-9])\b', change_year, text, count=1)
    
    def _mutate_number(self, text):
        def change_int(m):
            return str(int(int(m.group(0)) * self.rng.choice([0.5, 1.5, 2.0])))
        return re.sub(r'\b\d{3,6}\b', change_int, text, count=1)
        
    def _mutate_decimal(self, text):
        def change_float(m):
            val = float(m.group(0).replace(',', '.'))
            return str(round(val * self.rng.choice([1.3, 0.7, 2.1]), 1))
        return re.sub(r'\b\d+[.,]\d+\b', change_float, text, count=1)

    def _safe_replace_dict(self, text, replace_dict):
        
        for old, new in replace_dict.items():
            if ' ' in old or not old.strip().isalnum():
                if old in text:
                    return text.replace(old, new, 1)
        

        for old, new in replace_dict.items():
            if ' ' in old or not old.strip().isalnum():
                continue 
            pattern = r'\b' + re.escape(old.strip()) + r'\b'
            if re.search(pattern, text):
                return re.sub(pattern, new, text, count=1)
        return text

    def mutate(self, text):
        if pd.isna(text):
            return str(text), False
            
        original = text
        mutations = []
        
        if re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text):
            mutations.append(('year', self._mutate_year))
        if re.search(r'\b\d{3,6}\b', text):
            mutations.append(('int', self._mutate_number))
        if re.search(r'\b\d+[.,]\d+\b', text):
            mutations.append(('dec', self._mutate_decimal))
        if any(unit in text for unit in self.units):
            mutations.append(('unit', lambda t: self._safe_replace_dict(t, self.units)))
        if any(name in text for name in self.names):
            mutations.append(('name', lambda t: self._safe_replace_dict(t, self.names)))
        if any(place in text for place in self.geo):
            mutations.append(('geo', lambda t: self._safe_replace_dict(t, self.geo)))
        if any(word in text for word in self.negations):
            mutations.append(('neg', lambda t: self._safe_replace_dict(t, self.negations)))
        if any(word in text for word in self.dates):
            mutations.append(('date', lambda t: self._safe_replace_dict(t, self.dates)))
        if any(word in text for word in self.quantifiers):
            mutations.append(('quant', lambda t: self._safe_replace_dict(t, self.quantifiers)))
            
        if not mutations:
            return text, False, None
            
        mutation_type, mutate_func = self.rng.choice(mutations)
        mutated = mutate_func(text)
        
        if mutated != original:
            return mutated, True, mutation_type  
        return text, False, None
    
    
def apply_mutations(df_true, mutator):
    
    rows = []
    fail = 0
    mutation_st = {}

    for _, row in tqdm(df_true.iterrows(), total=len(df_true), desc="mutation"):
        mutated_text, success, m_type = mutator.mutate(row["model_answer"])
        
        if success:
            if m_type in mutation_st:
                mutation_st[m_type] += 1
            else:
                mutation_st[m_type] = 1
                
            rows.append({
                "prompt": row["prompt"],
                "model_answer": row["model_answer"],
                "is_hallucination": 0,
                "correct_answer": None,
                "comment": None
            })
            rows.append({
                "prompt": row["prompt"],
                "model_answer": mutated_text,
                "is_hallucination": 1,
                "correct_answer": None,
                "comment": "synthetic_fact_mutation"
            })
        else:
            fail += 1

    df_train = pd.DataFrame(rows)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_train, fail, mutation_st

def ensure_paired_dataset(df_mutated: pd.DataFrame) -> pd.DataFrame:
    mask = df_mutated.groupby("prompt")["is_hallucination"].transform("nunique") == 2
    return df_mutated[mask].reset_index(drop=True)


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)


def main(
    train_json: str = "data/raw/train.json",
    val_json: str = "data/raw/validation.json",
    output_csv: str = "data/train.csv",
    seed: int = 42
) -> None:
    random.seed(seed)
    
    df_raw = load_raw_data(train_json, val_json)
    df_true = filter_true_samples(df_raw)
    
    mutator = FactualMutator(seed=seed)
    df_mutated, fail_c, mut_stats = apply_mutations(df_true, mutator)
    
    df_paired = ensure_paired_dataset(df_mutated)
    
    save_dataset(df_paired, output_csv)
    print(f"outp: {output_csv}")
    print(f"shape: {df_paired.shape}")


if __name__ == "__main__":
    main()