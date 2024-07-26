import re
import sys
from num2words import num2words

class PLConfig:
    pass

class PolishReplacementConfig(PLConfig):
    SEP_MASK = '\x00'

    UNICODE_TO_ASCII = {
        'àáâãåāăąǟǡǻȁȃȧ': 'a',
        'æǣǽ': 'ae',
        'çćĉċč': 'c',
        'ďđ': 'd',
        'èéêëēĕėęěȅȇȩ': 'e',
        'ĝğġģǥǧǵ': 'g',
        'ĥħȟ': 'h',
        'ìíîïĩīĭįıȉȋ': 'i',
        'ĵǰ': 'j',
        'ķĸǩκ': 'k',
        'ĺļľŀł': 'l',
        'ñńņňŉŋǹ': 'n',
        'òóôõōŏőǫǭȍȏð': 'o',
        'œøǿ': 'oe',
        'ŕŗřȑȓ': 'r',
        'śŝşšș': 's',
        'ţťŧț': 't',
        'ùúûũūŭůűųȕȗ': 'u',
        'ŵ': 'w',
        'ýÿŷ': 'y',
        'źżžȥ': 'z',
    }



class PolishAcronymPhonemeConfig(PLConfig):
    EXCLUDE = {'EU', 'USA'}

    LETTER = {
        'A': 'a',
        'B': 'be',
        'C': 'ce',
        'D': 'de',
        'E': 'e',
        'F': 'ef',
        'G': 'ge',
        'H': 'ha',
        'I': 'i',
        'J': 'jot',
        'K': 'ka',
        'L': 'el',
        'M': 'em',
        'N': 'en',
        'O': 'o',
        'P': 'pe',
        'Q': 'ku',
        'R': 'er',
        'S': 'es',
        'T': 'te',
        'U': 'u',
        'V': 'fał',
        'W': 'wu',
        'X': 'iks',
        'Y': 'y',
        'Z': 'zet',
        'Ą': 'ą',
        'Ć': 'ć',
        'Ę': 'ę',
        'Ł': 'el',
        'Ń': 'ń',
        'Ś': 'ś',
        'Ź': 'ź',
        'Ż': 'ż',
    }

class PolishUnitConfig(PLConfig):
    PLURAL_NO_SUFFIX = {
        'kg': 'kilogram',
        'g': 'gram',
        'l': 'litr',
        'm': 'metr',
        'km': 'kilometr',
        'm2': 'metr kwadratowy',
        'm3': 'metr sześcienny',
        'ml': 'mililitr',
    }

    PLURAL_SUFFIX_N = {
        's': 'sekund',
        'min': 'minut',
        'h': 'godzin',
    }
    PLURAL_SUFFIX_EN = {
        'mln': 'milionów',
        'mld': 'miliardów',
        'tys': 'tysięcy',
    }

class PolishAbbreviationConfig(PLConfig):
    MISC = {
        'np.': 'na przykład',
        'itd.': 'i tak dalej',
        'dr': 'doktor',
        'mgr': 'magister',
    }

    TIME = {
        's': 'sekunda',
        'min': 'minuta',
        'h': 'godzina',
    }

    WEEKDAY = {
        'pon.': 'poniedziałek',
        'wt.': 'wtorek',
        'śr.': 'środa',
        'czw.': 'czwartek',
        'pt.': 'piątek',
        'sob.': 'sobota',
        'ndz.': 'niedziela',
    }

    MONTH = {
        'styczeń': 'styczeń',
        'luty': 'luty',
        'marzec': 'marzec',
        'kwiecień': 'kwiecień',
        'maj': 'maj',
        'czerwiec': 'czerwiec',
        'lipiec': 'lipiec',
        'sierpień': 'sierpień',
        'wrzesień': 'wrzesień',
        'październik': 'październik',
        'listopad': 'listopad',
        'grudzień': 'grudzień',
        'sty.': 'styczeń',
        'lut.': 'luty',
        'mar.': 'marzec',
        'kwi.': 'kwiecień',
        'cze.': 'czerwiec',
        'lip.': 'lipiec',
        'sie.': 'sierpień',
        'wrz.': 'wrzesień',
        'paź.': 'październik',
        'lis.': 'listopad',
        'gru.': 'grudzień',
    }

    NUMBER_MONTH = {
        '1': 'styczeń',
        '01': 'styczeń',
        '2': 'luty',
        '02': 'luty',
        '3': 'marzec',
        '03': 'marzec',
        '4': 'kwiecień',
        '04': 'kwiecień',
        '5': 'maj',
        '05': 'maj',
        '6': 'czerwiec',
        '06': 'czerwiec',
        '7': 'lipiec',
        '07': 'lipiec',
        '8': 'sierpień',
        '08': 'sierpień',
        '9': 'wrzesień',
        '09': 'wrzesień',
        '10': 'październik',
        '11': 'listopad',
        '12': 'grudzień',
    }
    CURRENCY_SYMBOL = {
        'PLN': 'złoty',
        'USD': 'dolar',
        'EUR': 'euro',
        'GBP': 'funt',
        'CHF': 'frank',
    }
    MATH_SYMBOL = {
        '+': 'plus',
        '-': 'minus',
        '*': 'razy',
        '/': 'podzielić przez',
        '=': 'równa się',
        '<': 'mniej niż',
        '>': 'więcej niż',
        '±': 'plus minus',
        '≤': 'mniejsze lub równe',
        '≥': 'większe lub równe',
        '^': 'do potęgi',
        '√': 'pierwiastek',
        '∑': 'suma',
        '∏': 'iloczyn',
        '%': 'procent',
    }


class RegExConfig(PLConfig):
    """
    CONFIG for various regular expressions used for transliterations
    """

    def __init__(self, generic_config, acronym_phoneme_config, unit_config, abbreviation_config):
        """
        Constructor

        :param generic_config:
        :param acronym_phoneme_config:
        :param unit_config:
        :param abbreviation_config:
        """
        try:
            self.generic_config = generic_config
            self.acronym_phoneme_config = acronym_phoneme_config
            self.unit_config = unit_config
            self.abbreviation_config = abbreviation_config

            self.SPECIAL_TRANSLITERATE = {
                '.*\+/\-.*': ('+/-', 'plus minus'),
                '.*&.*': ('&', ' i '),
                '(^|(?<=[\.!?;:\-\s]))([a-z]{0,1}|[\d]+)\s{0,1}[^-]\-\s{0,1}([\d]+|[a-z]{0,1})($|(?=[\.!?;:\-\s]+))': (
                    '-', ' do '),
                '\\b[02-9]+\s{0,1}/\s{0,1}\d+\\b': ('/', ' z '),
                '\\b1/10\\b': ('1/10', 'jedna dziesiąta'),
                '\\b⅒\\b': ('⅒', 'jedna dziesiąta'),
                '\\b1/9\\b': ('1/9', 'jedna dziewiąta'),
                '\\b⅑\\b': ('⅑', 'jedna dziewiąta'),
                '\\b1/8\\b': ('1/8', 'jedna ósma'),
                '\\b⅛\\b': ('⅛', 'jedna ósma'),
                '\\b1/7\\b': ('1/7', 'jedna siódma'),
                '\\b⅐\\b': ('⅐', 'jedna siódma'),
                '\\b1/6\\b': ('1/6', 'jedna szósta'),
                '\\b⅙\\b': ('⅙', 'jedna szósta'),
                '\\b1/5\\b': ('1/5', 'jedna piąta'),
                '\\b⅕\\b': ('⅕', 'jedna piąta'),
                '\\b1/4\\b': ('1/4', 'jedna czwarta'),
                '\\b¼\\b': ('¼', 'jedna czwarta'),
                '\\b1/3\\b': ('1/3', 'jedna trzecia'),
                '\\b⅓\\b': ('⅓', 'jedna trzecia'),
                '\\b1/2\\b': ('1/2', 'pół'),
                '\\b½\\b': ('½', 'pół'),
                '\\b1000\\b': ('1000', 'tysiąc'),
            }

            self.CURRENCY_MAGNITUDE = ['\\bmln\\b', '\\bmln\.\\b', '\\bmld\\b', '\\bmld\.\\b',
                                       '\\bmilion[y]{0,1}\\b', '\\bmiliard[y]{0,1}\\b',
                                       '\\btysiąc\\b']

            self.DETECT_ABBREVIATION = re.compile(
                '(^|(?<=[\.!?;:\-\s,\(\[\{]))([A-ZĄĆĘŁŃÓŚŹŻ]{2,}|([A-ZĄĆĘŁŃÓŚŹŻ]\.){2,})($|(?=[\.!?;:\-\s,\)\]\}]+))')
            self.DETECT_WEEKDAY = re.compile('\\b(' + '|'.join(self.abbreviation_config.WEEKDAY.keys()) + ')\\b')
            self.DETECT_MONTH = re.compile('\\b(' + '|'.join(self.abbreviation_config.MONTH.keys()) + ')\\b')
            self.DETECT_TIME_OF_DAY = re.compile(
                '(\\b(([0-1][0-9]|2[0-3])[\.\:]|[0-9][\.\:])([0-5][0-9]|[0-9])\s{0,1}(h|godz|godziny){0,1}\\b)')
            self.DETECT_TIMESTAMP = re.compile(
                '(\\b\d+(h|godz){0,1}:([0-5][0-9]|[0-9])(m|min){0,1}(:([0-5][0-9]|[0-9])(s|sek){0,1}){0,1}\\b)')
            self.DETECT_DATE = re.compile(
                '(([1-9]|(0[1-9])|(1[0-9])|(2[0-9])|30|31)\.(((([1-9]|0[1-9])|(10|11|12))\.)|(\s{0,1}('
                + '|'.join(self.abbreviation_config.MONTH.keys()) + ')(\.|\\b)))(\s{0,1}\d\d\d\d|\s{0,1}\d\d){0,1})')
            self.DETECT_ORDINAL = re.compile('[\(\[]{0,1}\d+\.[\)\]]{0,1}')
            self.DETECT_NUMBER = re.compile('([\+\-]{0,1}\d+[\d\.,]*)')
            self.DETECT_WHITESPACE_SEQ = re.compile('\s+')

            # static regex patterns but contents depending on input parameter config (thus set to None, initially)
            self.DETECT_CURRENCY_SYMBOL = None
            self.DETECT_CURRENCY_MAGNITUDE = None
            self.DETECT_CURRENCY = None

        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

class PolishTransliterate:
    def __init__(self,
                 transliterate_ops=['acronym_phoneme', 'accent_peculiarity', 'amount_money', 'date', 'timestamp',
                                    'time_of_day', 'ordinal', 'special'],
                 replace={'-': ' '},
                 sep_abbreviation=' ',
                 make_lowercase=True
                 ):
        self.transliterate_ops = transliterate_ops
        self.replace = replace
        self.sep_abbreviation = sep_abbreviation
        self.make_lowercase = make_lowercase

        self.generic_config = PolishReplacementConfig()
        self.acronym_phoneme_config = PolishAcronymPhonemeConfig()
        self.unit_config = PolishUnitConfig()
        self.abbreviation_config = PolishAbbreviationConfig()
        self.regex = RegExConfig(self.generic_config,
                                 self.acronym_phoneme_config,
                                 self.unit_config,
                                 self.abbreviation_config)

        escaped_cursym = [re.escape(it) for it in self.abbreviation_config.CURRENCY_SYMBOL.keys()]
        rstring_cursym_escaped = '|'.join(escaped_cursym).replace('_', self.generic_config.SEP_MASK)
        rstring_curmagn = '|'.join(self.regex.CURRENCY_MAGNITUDE)
        self.regex.DETECT_CURRENCY_SYMBOL = re.compile(rstring_cursym_escaped)
        self.regex.DETECT_CURRENCY_MAGNITUDE = re.compile(rstring_curmagn)

        cur_str = '(^|(?<=[\.!?;:\-\(\)\[\]\s]))(([\+\-]{0,1}\d+[\d\.,]*\s*(' \
                        + rstring_curmagn + '){0,1}\s*' + '(' \
                        + rstring_cursym_escaped + '))|' + '((' \
                        + rstring_cursym_escaped + ')\s*[\+\-]{0,1}\d+[\d\.,]*\s*(' \
                        + rstring_curmagn + '){0,1})|' + '(' \
                        + rstring_cursym_escaped \
                        + '))($|(?=[\.!?;:\-\(\)\[\]\s]+))'

        self.regex.DETECT_CURRENCY = re.compile(cur_str)

    def transliterate(self, text):
        # General note: Polish specific transformations go here

        text = self._mask_acronym(text)
        if 'acronym_phoneme' in self.transliterate_ops:
            text = self._acronym_phoneme_op(text)

        if self.make_lowercase:
            text = text.lower()

        if 'accent_peculiarity' in self.transliterate_ops:
            text = self._accent_peculiarity_op(text)

        if 'amount_money' in self.transliterate_ops:
            text = self._amount_money_op(text)

        if 'date' in self.transliterate_ops:
            text = self._date_op(text)

        if 'timestamp' in self.transliterate_ops:
            text = self._timestamp_op(text)

        if 'time_of_day' in self.transliterate_ops:
            text = self._timeofday_op(text)

        split_text = text.split(' ')
        cleaned_words = []

        idx = 0
        for word in split_text:
            if not word:
                continue

            for tr in self.transliterate_ops:
                if tr == 'weekday':
                    word = self._weekday_op(word)
                elif tr == 'month':
                    word = self._month_op(word)
                elif tr == 'ordinal':
                    word = self._ordinal_op(word, idx, split_text, cleaned_words)
                elif tr == 'special':
                    word = self._special_op(word)
                elif tr == 'math_symbol':
                    word = self._math_symbol_op(word)
                elif tr == 'spoken_symbol':
                    word = self._spoken_symbol_op(word, idx, split_text)

            for old, new in self.replace.items():
                word = word.replace(old, new)

            word = self._misc_abbreviation_op(word)
            word = self._number_unit_op(word, idx, cleaned_words)

            cleaned_words.append(word)
            idx += 1

        text = ' '.join(cleaned_words).replace(self.generic_config.SEP_MASK, self.sep_abbreviation)
        return self.regex.DETECT_WHITESPACE_SEQ.sub(' ', text)

    def _mask_acronym(self, text):
        try:
            abbr_expanded = []
            for abbr in self.regex.DETECT_ABBREVIATION.finditer(text):
                if abbr.group(0) not in self.acronym_phoneme_config.EXCLUDE:
                    abbr_expanded.append(
                        (abbr.group(0), self.generic_config.SEP_MASK.join([c for c in
                            abbr.group(0).replace('.', '')
                        ])))
            for m in abbr_expanded:
                text = text.replace(m[0], m[1], 1)

            return text
        except Exception as e:
            raise e

    def _acronym_phoneme_op(self, text):
        try:
            abbr_expanded = []
            for abbr in self.regex.DETECT_ABBREVIATION.finditer(text):
                if abbr.group(0) not in self.acronym_phoneme_config.EXCLUDE:
                    abbr_expanded.append(
                        (abbr.group(0), self.generic_config.SEP_MASK.join([
                            self.acronym_phoneme_config.LETTER[c] for c in
                            abbr.group(0).replace('.', '')
                        ])))
            for m in abbr_expanded:
                text = text.replace(m[0], m[1], 1)

            return text
        except Exception as e:
            raise e

    def _accent_peculiarity_op(self, text):
        try:
            for chars, mapped in self.generic_config.UNICODE_TO_ASCII.items():
                text = re.sub('|'.join([c for c in chars]), mapped, text)

            return text
        except Exception as e:
            raise e

    def _amount_money_op(self, text):
        try:
            diff_len = 0
            for mc in self.regex.DETECT_CURRENCY.finditer(text):
                match_currency = self._acronym_phoneme_op(mc.group(0))
                mc_end = mc.end()

                m_symbol = self.regex.DETECT_CURRENCY_SYMBOL.search(match_currency)
                m_magnitude = self.regex.DETECT_CURRENCY_MAGNITUDE.search(match_currency)
                m_number = self.regex.DETECT_NUMBER.search(match_currency)

                number = m_number.group(0) if m_number else ''
                if not m_magnitude and ',' in number:
                    cur_symbol = self.abbreviation_config.CURRENCY_SYMBOL[
                                                        m_symbol.group(0).replace(self.generic_config.SEP_MASK, '_')]
                    number = number.replace(',', ' ' + cur_symbol + ' ')
                    dec_start = number.rfind(' ' + cur_symbol)+len(cur_symbol)+2
                    decimals = number[dec_start:]
                    if int(decimals) == 0:
                        number = number[:dec_start]
                    elif len(decimals) > 2:
                        number = number[:dec_start] + ' ' + decimals[0:2] + ' ' + ' '.join(decimals[2:])
                    rearranged_currency_term = number
                else:
                    rearranged_currency_term = number + ' ' if m_number else ''
                    rearranged_currency_term += m_magnitude.group(0) + ' ' if m_magnitude else ''
                    rearranged_currency_term += self.abbreviation_config.CURRENCY_SYMBOL[
                        m_symbol.group(0).replace(self.generic_config.SEP_MASK, '_')]
                text = text[:mc.start() + diff_len] + rearranged_currency_term + text[mc_end + diff_len:]
                diff_len = len(rearranged_currency_term) - (mc_end - (mc.start() + diff_len))

            return text
        except Exception as e:
            raise e

    def _date_op(self, text):
        try:
            for date_m in self.regex.DETECT_DATE.finditer(text):
                frags = date_m.group(0).split('.')
                if ' ' in frags[-1]:
                    space_split = frags[-1].strip().split(' ')
                    del (frags[-1])
                    frags.extend(space_split)
                day = num2words(frags[0], lang='pl', to='ordinal')
                if date_m.start() > 1 and text[date_m.start() - 2:date_m.start()] in ('m ', 'n '):
                    day += 'n'
                if frags[1].strip() in self.abbreviation_config.MONTH:
                    month = self.abbreviation_config.MONTH[frags[1].strip()]
                else:
                    month = self.abbreviation_config.NUMBER_MONTH[frags[1].strip()]
                year = ''
                if len(frags) == 3 and frags[2]:
                    year = num2words(frags[2], lang='pl', to='year')
                text = self.regex.DETECT_DATE.sub(day + ' ' + month + (' ' + year if year else ''), text, count=1)

            return text
        except Exception as e:
            raise e

    def _timestamp_op(self, text):
        try:
            for timestamp_m in self.regex.DETECT_TIMESTAMP.finditer(text):
                ts = timestamp_m.group(0)
                ts_split = ts.split(':')
                if len(ts_split) == 2:
                    if int(ts_split[0].replace('h', '').replace('std', '')) == 1:
                        ts = 'jedna godzina '
                    else:
                        ts = ts_split[0] + ' godzin '
                    if int(ts_split[1].replace('m', '').replace('min', '')) == 1:
                        ts += 'jedna minuta'
                    else:
                        ts += ts_split[1] + ' minut'
                else:
                    if int(ts_split[0].replace('h', '').replace('std', '')) == 1:
                        ts = 'jedna godzina '
                    else:
                        ts = ts_split[0] + ' godzin '
                    if int(ts_split[1].replace('min', '').replace('m', '')) == 1:
                        ts += 'jedna minuta '
                    else:
                        ts += ts_split[1].replace('min', '').replace('m', '') + ' minut '
                    if int(ts_split[2].replace('sek', '').replace('sec', '').replace('s', '')) == 1:
                        ts += 'jedna sekunda'
                    else:
                        ts += ts_split[2].replace('sek', '').replace('sec', '').replace('s', '') + ' sekund'

                text = text[:timestamp_m.start()] + ts + text[timestamp_m.end():]

            return text
        except Exception as e:
            raise e

    def _timeofday_op(self, text):
        try:
            for time_m in self.regex.DETECT_TIME_OF_DAY.finditer(text):
                tod = text[time_m.start():time_m.end()].replace('uhr', '').replace('h', '').replace(':',
                                                                                                    ' godzina ').replace(
                    '.', ' godzina ')
                if int(tod.split(' godzina ')[0]) == 1:
                    tod = 'jedna godzina ' + tod.split(' godzina ')[1]
                text = text[:time_m.start()] + tod + text[time_m.end():]

            return text
        except Exception as e:
            raise e

    def _weekday_op(self, word):
        try:
            ms = self.regex.DETECT_WEEKDAY.finditer(word)
            offset = 0
            for m in ms:
                _word = word[m.start() + offset:m.end() + offset].replace('.', '')
                if _word in self.abbreviation_config.WEEKDAY:
                    word = word[:m.start() + offset] + self.abbreviation_config.WEEKDAY[_word] + word[m.end() + offset:]
                    offset += len(self.abbreviation_config.WEEKDAY[_word]) - (m.end() - m.start())

            if offset > 0:
                return word.replace('.', '')
            else:
                return word

        except Exception as e:
            raise e

    def _month_op(self, word):
        try:
            ms = self.regex.DETECT_MONTH.finditer(word)
            offset = 0
            for m in ms:
                _word = word[m.start() + offset:m.end() + offset].replace('.', '')
                if _word in self.abbreviation_config.MONTH:
                    word = word[:m.start() + offset] + self.abbreviation_config.MONTH[_word] + word[m.end() + offset:]
                    offset += len(self.abbreviation_config.MONTH[_word]) - (m.end() - m.start())

            if offset > 0:
                return word.replace('.', '')
            else:
                return word

        except Exception as e:
            raise e

    def _ordinal_op(self, word, idx, split_text, cleaned):
        try:
            if self.regex.DETECT_ORDINAL.match(word) and word.endswith('.'):
                if idx < (len(split_text) - 1) \
                        and split_text[idx + 1] not in self.abbreviation_config.CURRENCY_SYMBOL.values():
                    word = num2words(word, lang='pl', to='ordinal')
                    if idx > 0 and idx < (len(split_text) - 1):
                        if cleaned[idx - 1].endswith('m'):
                            word += 'n'

            return word

        except Exception as e:
            raise e

    def _special_op(self, word):
        try:
            for pat, tup_repl in self.regex.SPECIAL_TRANSLITERATE.items():
                if re.search(pat, word):
                    word = word.replace(tup_repl[0], tup_repl[1], 1)
                    ws = []
                    for w in word.split(' '):
                        if self.regex.DETECT_NUMBER.match(w):
                            ws.append(self._transliterate_number(w))
                        else:
                            ws.append(w)
                    if ws:
                        word = ' '.join(ws)

            return word
        except Exception as e:
            raise e

    def _math_symbol_op(self, word):
        try:
            for pat, repl in self.abbreviation_config.MATH_SYMBOL.items():
                if pat in word:
                    if pat == 'x':
                        if len(word) > 1 and word.find(pat) == 0:
                            continue
                        elif len(word) > 1 and not word[:word.find(pat)].isdecimal():
                            continue
                    elif pat == '-':
                        if word == '--':
                            continue
                        elif len(word) > 1 and \
                                (len(word[:word.find(pat)]) > 1 or len(word[word.find(pat) + 1:]) > 1):
                            continue

                    word = word.replace(pat, repl, 1)

            return word
        except Exception as e:
            raise e

    def _spoken_symbol_op(self, word, idx, split_text):
        try:
            for pats, repl in self.abbreviation_config.SPOKEN_SYMBOL.items():
                if pats[0] in word:
                    word = word.replace(pats[0], repl.replace('_', self.generic_config.SEP_MASK))
                    if pats[1] in word:
                        word = word.replace(pats[1], self.generic_config.SEP_MASK)
                    else:
                        for fwd_idx in range(idx + 1, len(split_text)):
                            if pats[1] in split_text[fwd_idx]:
                                split_text[fwd_idx] = split_text[fwd_idx].replace(pats[1],
                                                                                  self.generic_config.SEP_MASK)
                                break

            return word
        except Exception as e:
            raise e

    def _misc_abbreviation_op(self, word):
        try:
            for short, long in self.abbreviation_config.MISC.items():
                if long not in word:
                    if (short + '.') == word or short == word:
                        word = long

            return word
        except Exception as e:
            raise e

    def _number_unit_op(self, word, idx, cleaned_words):
        try:
            w_unit = word
            if w_unit.endswith('.'):
                w_unit = w_unit[:-1]
            w_unit = w_unit.replace(self.generic_config.SEP_MASK, '_')

            if w_unit in self.unit_config.PLURAL_NO_SUFFIX.keys():
                word = self.unit_config.PLURAL_NO_SUFFIX[w_unit]
                if idx > 0 and cleaned_words[idx - 1] == ('jeden'):
                    cleaned_words[idx - 1] = 'jedna'
            elif w_unit in self.unit_config.PLURAL_SUFFIX_EN.keys():
                word = self.unit_config.PLURAL_SUFFIX_EN[w_unit]
                if idx > 0 and (cleaned_words[idx - 1] == ('jeden')
                                or cleaned_words[idx - 1] == ('jedna')
                                or cleaned_words[idx - 1] == ('jedną')):
                    word = word[:-2]
                    cleaned_words[idx - 1] = 'jedna'

            elif w_unit in self.unit_config.PLURAL_SUFFIX_N.keys():
                word = self.unit_config.PLURAL_SUFFIX_N[w_unit]
                if idx > 0 and (cleaned_words[idx - 1] == ('jeden')
                                or cleaned_words[idx - 1] == ('jedna')
                                or cleaned_words[idx - 1] == ('jedną')):
                    word = word[:-1]
                    cleaned_words[idx - 1] = 'jedna'

            num_match = self.regex.DETECT_NUMBER.match(word)
            if num_match:

                w_unit = word[num_match.end():]
                invariant_one = False
                if w_unit:
                    if w_unit.endswith('.'):
                        w_unit = w_unit[:-1]

                    if w_unit in self.unit_config.PLURAL_NO_SUFFIX.keys():
                        w_unit = self.unit_config.PLURAL_NO_SUFFIX[w_unit]
                    elif w_unit in self.unit_config.PLURAL_SUFFIX_EN.keys():
                        w_unit = self.unit_config.PLURAL_SUFFIX_EN[w_unit]
                        if num_match.group(0).replace('+', '').replace('-', '') in ['1', '1.0', '1.00']:
                            w_unit = w_unit[:-2]
                            invariant_one = True
                    elif w_unit in self.unit_config.PLURAL_SUFFIX_N.keys():
                        w_unit = self.unit_config.PLURAL_SUFFIX_N[w_unit]
                        if num_match.group(0).replace('+', '').replace('-', '') in ['1', '1.0', '1.00']:
                            w_unit = w_unit[:-1]
                            invariant_one = True
                    w_unit = ' ' + w_unit
                    word = num_match.group(0)

                if invariant_one:
                    word = 'jedna'
                else:
                    word = self._transliterate_number(word)
                word += w_unit

            return word
        except Exception as e:
            raise e

    def _transliterate_number(self, number: str) -> str:
        try:
            if number.count(',') == 1 and number.count('.') >= 1:
                number = number.replace('.', '')

            try:
                if number.count(',') == 1:
                    number = number.replace(',', '.')
                    word = num2words(float(number), lang='pl', to='cardinal').lower()
                elif number.count('.') >= 1:
                    number = number.replace('.', '')
                    word = num2words(int(number), lang='pl', to='cardinal').lower()
                else:
                    word = num2words(int(number), lang='pl', to='cardinal').lower()
            except ValueError:
                word = number
            return word
        except Exception as e:
            raise e


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('ERROR: No text given')
        sys.exit(-1)

    ops = {'accent_peculiarity', 'amount_money', 'date', 'timestamp', 'time_of_day', 'ordinal', 'special'}

    text = sys.argv[1]
    normalized_text = PolishTransliterate(transliterate_ops=ops).transliterate(text)
    print(normalized_text)