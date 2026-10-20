# -*- coding: utf-8 -*-
"""Custom phonetic Tamil to Thanglish transliterator for natural conversational Thanglish."""

# Base consonants (மெய்யெழுத்துக்கள்)
MEI = {
    'க்': 'k',
    'ங்': 'ng',
    'ச்': 'ch',
    'ஞ்': 'nj',
    'ட்': 't',
    'ண்': 'n',
    'த்': 'th',
    'ந்': 'n',
    'ப்': 'p',
    'ம்': 'm',
    'ய்': 'y',
    'ர்': 'r',
    'ல்': 'l',
    'வ்': 'v',
    'ழ்': 'zh',
    'ள்': 'l',
    'ற்': 'r',
    'ன்': 'n',
    'ஜ்': 'j',
    'ஷ்': 'sh',
    'ஸ்': 's',
    'ஹ்': 'h',
    'க்ஷ்': 'ksh',
    'ஸ்ரீ': 'sri'
}

# Standalone vowels (உயிரெழுத்துக்கள்)
UYIR = {
    'அ': 'a',
    'ஆ': 'aa',
    'இ': 'i',
    'ஈ': 'ee',
    'உ': 'u',
    'ஊ': 'oo',
    'எ': 'e',
    'ஏ': 'ae',
    'ஐ': 'ai',
    'ஒ': 'o',
    'ஓ': 'oa',
    'ஔ': 'au',
    'ஃ': 'akh'
}

# Vowel markers corresponding to uyirmei
UYIR_SUFFIXES = [
    ('', 'a'),
    ('ா', 'aa'),
    ('ி', 'i'),
    ('ீ', 'ee'),
    ('ு', 'u'),
    ('ூ', 'oo'),
    ('ெ', 'e'),
    ('ே', 'ae'),
    ('ை', 'ai'),
    ('ொ', 'o'),
    ('ோ', 'oa'),
    ('ௌ', 'au'),
]

CONSONANTS = [
    ('க்', 'k'),
    ('ங்', 'ng'),
    ('ச்', 'ch'),
    ('ஞ்', 'nj'),
    ('ட்', 't'),
    ('ண்', 'n'),
    ('த்', 'th'),
    ('ந்', 'n'),
    ('ப்', 'p'),
    ('ம்', 'm'),
    ('ய்', 'y'),
    ('ர்', 'r'),
    ('ல்', 'l'),
    ('வ்', 'v'),
    ('ழ்', 'zh'),
    ('ள்', 'l'),
    ('ற்', 'r'),
    ('ன்', 'n'),
    ('ஜ்', 'j'),
    ('ஷ்', 'sh'),
    ('ஸ்', 's'),
    ('ஹ்', 'h'),
    ('க்ஷ்', 'ksh'),
]

# Pull the base consonant glyph without virama
PURE_CONSONANTS_MAP = {
    'க': 'k', 'ங': 'ng', 'ச': 'ch', 'ஞ': 'nj', 'ட': 't', 'ண': 'n',
    'த': 'th', 'ந': 'n', 'ப': 'p', 'ம': 'm', 'ய': 'y', 'ர': 'r',
    'ல': 'l', 'வ': 'v', 'ழ': 'zh', 'ள': 'l', 'ற': 'r', 'ன': 'n',
    'ஜ': 'j', 'ஷ': 'sh', 'ஸ': 's', 'ஹ': 'h', 'க்ஷ': 'ksh'
}

TABLE = {}
TABLE.update(UYIR)
TABLE.update(MEI)

# Populate uyirmei
for c_char, c_rom in PURE_CONSONANTS_MAP.items():
    # Base with implicit 'a'
    TABLE[c_char] = c_rom + 'a'
    # Base with vowel signs
    TABLE[c_char + 'ா'] = c_rom + 'aa'
    TABLE[c_char + 'ி'] = c_rom + 'i'
    TABLE[c_char + 'ீ'] = c_rom + 'ee'
    TABLE[c_char + 'ு'] = c_rom + 'u'
    TABLE[c_char + 'ூ'] = c_rom + 'oo'
    TABLE[c_char + 'ெ'] = c_rom + 'e'
    TABLE[c_char + 'ே'] = c_rom + 'ae'
    TABLE[c_char + 'ை'] = c_rom + 'ai'
    TABLE[c_char + 'ொ'] = c_rom + 'o'
    TABLE[c_char + 'ோ'] = c_rom + 'oa'
    TABLE[c_char + 'ௌ'] = c_rom + 'au'

# Custom colloquial adjustments
SPECIAL_WORDS = {
    'வணக்கம்': 'vanakkam',
    'நான்': 'naan',
    'ஒரு': 'oru',
    'மாணவன்': 'maanavan',
    'வீட்டுக்கு': 'veetuku',
    'வீட்டுக்குப்': 'veetuku',
    'போகிறேன்': 'poren',
    'போறேன்': 'poren',
    'எப்படி': 'eppadi',
    'இருக்கிறீர்கள்': 'irukkeenga',
    'நன்றாக': 'nandraga',
    'இருக்கிறேன்': 'irukken',
    'உங்கள்': 'ungal',
    'பெயர்': 'peyar',
    'என்ன': 'enna',
    'சென்னையில்': 'chennaiyil',
    'படிக்கிறேன்': 'padikkiren',
    'மிகவும்': 'migavum',
}

def to_thanglish(text: str) -> str:
    import re
    tokens = re.split(r'(\s+|[.,!?:;\"\'\(\)\-])', text)
    out = []
    for tok in tokens:
        if not tok:
            continue
        # Check special common words
        clean_tok = tok.strip()
        if clean_tok in SPECIAL_WORDS:
            out.append(SPECIAL_WORDS[clean_tok])
            continue
        # Character-level transliteration for Tamil
        import tamil
        letters = tamil.utf8.get_letters(tok)
        tok_out = []
        for l in letters:
            tok_out.append(TABLE.get(l, l))
        out.append("".join(tok_out))
    
    res = "".join(out)
    # Naturalizing common patterns
    res = res.replace("kk", "kk").replace("thth", "tth")
    return res

if __name__ == "__main__":
    t1 = "வணக்கம்"
    t2 = "நான் ஒரு மாணவன்"
    t3 = "நான் வீட்டுக்குப் போகிறேன்."
    print(t1, "->", to_thanglish(t1))
    print(t2, "->", to_thanglish(t2))
    print(t3, "->", to_thanglish(t3))
