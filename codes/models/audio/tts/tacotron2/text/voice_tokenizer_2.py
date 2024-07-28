import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from pl_transliterate import PolishTransliterate


def text_cleaners(text):
  text = PolishTransliterate().transliterate(text)
  text = text.lower()
  text = text.sub(re.compile(r'\s+'), ' ', text)
  text = text.replace('"', '')
  return text

def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(', '}': ')',
        '[': '(', ']': ')',
        '`': '\'', '—': '-',
        '—': '-', '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word

with open(f'../../../../../../dataset/transcriptions.txt', 'r', encoding='utf-8') as at:
    ttsd = at.readlines()

allowed_characters_re = re.compile(r'^[a-ząćęłńóśźż!:;"/, \-\(\)\.\'\?ʼ]+$')
def preprocess_word(word, report=False):
    word = text_cleaners(word)
    word = remove_extraneous_punctuation(word)
    if not bool(allowed_characters_re.match(word)):
        if report and word:
            print(f"REPORTING: '{word}'")
        return ''
    return word

def batch_iterator(batch_size=1000):
    print("Processing ASR texts.")
    for i in range(0, len(ttsd), batch_size):
        yield [preprocess_word(t, True) for t in ttsd[i:i+batch_size]]

print("Training...")
trainer = BpeTrainer(special_tokens=['[STOP]', '[UNK]', '[SPACE]'], vocab_size=255)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
print("pre_tokenizer done")
print("train_from_iterator....")
tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ttsd))#+len(bcd))
print("train_from_iterator done")

# print(tokenizer.decode(tokenizer.encode("podróżowałem tu i tam 112 lat temu{{}}").ids))

tokenizer.save('../../../../../../dataset/polish_tokenizer.json')
