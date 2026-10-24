import insertions as e
from ngram_overlap import SourceIndex,tokenize

e.init();checks=[]
def check(name,condition):
 assert condition,name
 checks.append(name)
examples=[
 ('The authors report the samples were collected from healthy volunteers.','reporting_complementizer_that','grammar_insert'),
 ('The model we used for this analysis produced stable estimates.','object_relative_that','grammar_insert'),
 ('All the samples were collected from healthy volunteers.','quantifier_of','grammar_insert'),
 ('The signal was measured using an optical detector in this experiment.','method_by_using','grammar_insert'),
 ('The instrument was used to measure the response in each sample.','purpose_in_order','grammar_insert'),
 ('The treatment was applied for five days in this experiment.','explicit_duration_period','grammar_insert'),
 ('The samples were incubated at 25 °C for the experiment.','explicit_temperature','grammar_insert'),
 ('The measurements were collected in 2020 from several laboratories.','explicit_calendar_year','grammar_insert'),
 ('This study evaluates several methods for estimating the response.','deictic_particular','expanded_insert'),
 ('Several methods such as optical spectroscopy are used in these experiments.','redundant_example_cue','expanded_insert')]
for text,rule,mode in examples:
 candidates=e.candidates(text,mode);check('candidate '+rule,any(c['rule']==rule for c in candidates))
 after,edits=e.repair(text,SourceIndex(text),mode)
 check('repair '+rule,after!=text and e.verify_trace(text,after,edits))
 check('scientific guard '+rule,e.guard(text,after))
for text in ['The authors report that the samples were collected from healthy volunteers.', 'The patients using this drug were enrolled in the trial.', 'The model failed to predict the observed result.', 'The signal $x = 5$ was the main result.']:
 check('no unsafe candidate '+text,not e.candidates(text,'grammar_insert'))
for text in ['All that is required is a stable detector.', 'The results were all the more convincing.', 'All the same, the model failed.', 'The effect was seen in 2000 patients.']:
 check('exclude idiom/count-as-year '+text,not e.candidates(text,'grammar_insert'))
check('preserve uncertainty',not e.guard('The model may predict the result.','The model predicts the result.'))
check('preserve negation',not e.guard('The model did not predict the result.','The model did predict the result.'))
check('preserve numbers',not e.guard('The sample contains 5 mg.','The sample contains 50 mg.'))
text='The model predicts the observed response in every sample.'
a=e.covered_positions(tokenize(text)[1],SourceIndex(text))[1];b=e.covered_positions(tokenize(text+' in other words')[1],SourceIndex(text))[1]
check('denominator-only control has no absolute gain',len(a)==len(b))
print('PASS',len(checks),'insertion/protection/denominator checks')
