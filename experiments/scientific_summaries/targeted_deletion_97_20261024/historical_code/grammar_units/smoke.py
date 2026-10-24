"""Small deterministic guard checks, not a substitute for the 97-paper audit."""
import grammar_units as g
import number_style as n
from ngram_overlap import SourceIndex

g.init()
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)

for text in ['LAXPC 20 data reveal the periodic signal in the observations.', 'NuSTAR 20 data reveal the periodic signal in the observations.', 'GPT 4 models have clear limitations in this experiment.']:
    check('identifier protected: '+text,not n.candidates(text,'numbers_percent_guarded'))
check('ordinary counts remain eligible',bool(n.candidates('The experiment included 20 participants in each treatment group.','numbers_percent_guarded')))
for text in ['The data show that the treatment reduces the observed response.', 'The authors report that it may reduce the response.', 'The experiments indicate that these results are robust.']:
    c=list(g.optional_that_candidates(text));check('optional that: '+text,len(c)==1)
    out,edits=g.repair(text,SourceIndex(text),'optional_that');check('grammar edit/provenance',out!=text and g.verify_trace(text,out,edits))
for text in ['The data show that result in the last panel.', 'The result that the authors report is preliminary.', 'We require that the model be stable.', 'The instrument is calibrated so that drift is minimal.']:
    check('retain obligatory/demonstrative that: '+text,not list(g.optional_that_candidates(text)))
for text in ['The sample contained 20 mg of the tested compound.', 'The sample contained twenty milligrams of the tested compound.', 'The recording lasted 1 ms in the current experiment.', 'The samples contained 2 μg of the tested compound.']:
    out,edits=g.repair(text,SourceIndex(text),'unit_aliases')
    check('unit roundtrip: '+text,out!=text and g.verify_trace(text,out,edits) and g.invariant(text,out))
for text in ['The concentration was 20 mg/mL in all samples.', 'The relationship was $x = 20 mg$ in this notation.', 'The sample was 20 MG in this notation.', 'The measure was 20 m in the current experiment.']:
    check('unit formula/ambiguity guard: '+text,not list(g.unit_candidates(text)))
# Neither a value change nor a change in unit magnitude is equivalent.
check('reject changed quantity',not g.invariant('The mass was 20 mg.','The mass was 200 milligrams.'))
check('reject changed unit scale',not g.invariant('The mass was 20 mg.','The mass was 20 grams.'))
print('PASS',len(checks),'guard/provenance checks')
