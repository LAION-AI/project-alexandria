"""Conservative number, unit and mathematical-notation preservation checks.

Canonical equivalence handles spacing, Unicode/LaTeX notation, decimal trailing
zeros and scientific notation. A failed check means review, not a proven error.
It intentionally does not claim general algebraic or unit-conversion equivalence.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import re
import unicodedata

SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾', '0123456789+-=()')
SUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎', '0123456789+-=()')
GREEK = {'alpha':'α','beta':'β','gamma':'γ','delta':'δ','epsilon':'ε','theta':'θ',
         'lambda':'λ','mu':'μ','nu':'ν','pi':'π','rho':'ρ','sigma':'σ','tau':'τ',
         'phi':'φ','omega':'ω','Delta':'Δ','Sigma':'Σ','Omega':'Ω'}
NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)')
RAW_NUMBER = re.compile(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?')
SCI = re.compile(r'([-+]?\d+(?:\.\d+)?)\s*[*]\s*10\s*\^\s*([-+]?\d+)')
UNITS = {'percent':'%','percentage':'%','metres':'m','meters':'m','metre':'m','meter':'m',
         'seconds':'s','second':'s','sec':'s','secs':'s','milliseconds':'ms','millisecond':'ms',
         'grams':'g','gram':'g','kilograms':'kg','kilogram':'kg','kelvin':'K','Kelvin':'K',
         'degrees':'degrees','patients':'patients','participants':'participants'}
UNIT_PATTERN = re.compile(r'^\s*(?:[-–]\s*)?(%|°[CF]|[μnmkMG]?eV|[μnmcdk]?m(?:\^\{?[-+]?\d+\}?)?|'
                          r'[μnmck]?g|[μnm]?s|Hz|kHz|MHz|GHz|Pa|kPa|MPa|GPa|K|W|kW|'
                          r'J|kJ|mL|μL|L|mol|mmol|M|mM|μM|percent|percentage|metres?|meters?|'
                          r'seconds?|secs?|milliseconds?|grams?|kilograms?|[Kk]elvin)(?![\w])')
VAR = r'(?:[A-Za-zα-ωΑ-Ω](?:_\{[\w+\-]+\}|_[\w]+)?(?:\^\{?[-+\d]+\}?)?)'
NUM = r'(?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
ATOM = '(?:' + VAR + '|' + NUM + ')'
RELATION = re.compile(ATOM + r'\s*(?:<=|>=|!=|[=<>≤≥≠≈])\s*' + ATOM)
OPERATION = re.compile(ATOM + r'\s*[+*/^]\s*' + ATOM)
LATEX = re.compile(r'\$\$.*?\$\$|\$[^$\n]+\$|\\\(.*?\\\)|\\\[.*?\\\]', re.S)
CHEMICAL = re.compile(r'\b(?:[A-Z][a-z]?\d*){1,8}\b')


def notation(text):
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]+', lambda m:'^{'+m.group().translate(SUP)+'}', text)
    text = re.sub(r'[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]+', lambda m:'_{'+m.group().translate(SUB)+'}', text)
    text = unicodedata.normalize('NFKC', text).replace('−','-').replace('×','*').replace('·','*')
    text = re.sub(r'\\(?:mathrm|text|mathbf|mathit)\{([^{}]*)\}', r'\1', text)
    for name, symbol in GREEK.items():
        text = re.sub(r'\\'+name+r'\b', lambda m:symbol, text)
    for old, new in [(r'\cdot','*'),(r'\times','*'),(r'\leq','≤'),(r'\geq','≥'),
                     (r'\neq','≠'),(r'\approx','≈'),(r'\pm','±')]:
        text = text.replace(old,new)
    text = re.sub(r'(?<=\d)[–-](?=\d)', ' to ', text)
    text = re.sub(r'\^\{([^{}]+)\}', r'^\1', text)
    text = re.sub(r'_\{([\w+\-]+)\}', r'_\1', text)
    text = SCI.sub(lambda m:m.group(1)+'e'+m.group(2), text)
    text = re.sub(r'(?<=\d)(?=(?:mm|cm|nm|μm|km|m|ms|μs|s|kg|mg|μg|g|MeV|GeV|eV|K|Hz)\b)', ' ', text)
    return text


def decimal(value):
    try:
        number = Decimal(value.replace(',',''))
        return str(number.normalize()) if number else '0'
    except InvalidOperation:
        return value


def canonical_math(text):
    text = notation(text)
    text = text.replace('\\(','').replace('\\)','').replace('\\[','').replace('\\]','').replace('$','')
    text = re.sub(r'\s+', '', text)
    text = text.replace('<=','≤').replace('>=','≥').replace('!=','≠')
    return RAW_NUMBER.sub(lambda m:decimal(m.group()), text)


def signature(text):
    normalized = notation(text)
    matches = list(NUMBER.finditer(normalized))
    numbers = [decimal(m.group()) for m in matches]
    quantities = []
    for match in matches:
        unit = UNIT_PATTERN.match(normalized[match.end():])
        if unit:
            value = unit.group(1)
            quantities.append((decimal(match.group()), UNITS.get(value,value)))
    formulas = [canonical_math(m.group()) for m in LATEX.finditer(text)]
    formulas += [canonical_math(m.group()) for pattern in [RELATION,OPERATION] for m in pattern.finditer(normalized)]
    formulas += ['chemical:'+m.group() for m in CHEMICAL.finditer(normalized) if any(c.isdigit() for c in m.group())]
    variables = re.findall(r'[α-ωΑ-Ω]|[A-Za-z]_\{[\w+\-]+\}|[A-Za-z]_[\w]+', normalized)
    operators = re.findall(r'<=|>=|!=|[<>≤≥≠±≈]', normalized)
    return dict(numbers=numbers, quantities=quantities, formulas=sorted(formulas),
                mathematical_variables=sorted(variables), operators=operators)


def check(before, after):
    old, new = signature(before), signature(after)
    flags = {key:old[key]==new[key] for key in old}
    return dict(passed=all(flags.values()), checks=flags, before=old, after=new,
                numeric_content_present=bool(old['numbers']),
                formula_content_present=bool(old['formulas'] or old['mathematical_variables']),
                exact_text_unchanged=before==after,
                interpretation='Conservative preservation filter; changed signatures require review and can include harmless reordering or equivalent unit/algebra conversions')
