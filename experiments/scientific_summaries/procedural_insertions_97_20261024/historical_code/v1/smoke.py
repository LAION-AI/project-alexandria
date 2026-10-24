import insertions as e
from ngram_overlap import SourceIndex,tokenize

e.init();checks=[]
def check(name,condition):
 assert condition,name
 checks.append(name)
source='Magnetic resonance imaging (MRI) was used for the scans. The imaging protocol was stable.'
check('source-defined acronym',e.source_glossary(source).get('MRI')=='Magnetic resonance imaging')
check('ambiguous acronym rejected','MRI' not in e.source_glossary(source+' Model resolution index (MRI) was another measure.'))
examples=[
 ('The sample contained 5 mg of the tested compound.','unit_name_gloss',None),
 ('The recording lasted 20 ms for every sample.','unit_name_gloss',None),
 ('The MRI data were evaluated using the same protocol.','source_acronym_gloss',source),
 ('The experiment was conducted at 2 atm for the test.','typed_pressure',None),
 ('The signal was recorded at 5 GHz in the experiment.','typed_frequency',None),
 ('The particles were measured at 5 eV in the experiment.','typed_energy',None),
 ('The instrument was used for measuring the response.','purpose_for_gerund',None),
 ('The method was designed to measure the signal.','purpose_design_in_order',None),
 ('The result was obtained by averaging the observations.','gerund_by_elaboration',None),
 ('The result was obtained through averaging the observations.','gerund_through_elaboration',None),
 ('The samples were prepared before heating the mixture.','gerund_before_elaboration',None),
 ('The samples were measured after heating the mixture.','gerund_after_elaboration',None),
 ('The detector was stable during the experiment.','during_course',None),
 ('The signal was measured with a detector in the experiment.','instrument_with',None),
 ('The signal was analyzed via a software package in the experiment.','instrument_via',None),
 ('Both these samples were evaluated in the experiment.','both_of_plural_np',None),
 ('The results shown in Table 1 support the hypothesis.','textual_reduced_relative',None),
 ('The samples collected from the trial were evaluated yesterday.','past_reduced_relative',None),
 ('The samples were incubated for 10 min before measurement.','abbreviated_duration',None),
]
for text,rule,extra in examples:
 index=SourceIndex(extra or text)
 check('candidate '+rule,any(c['rule']==rule for c in e.candidates(text,index)))
check('no orbital unit gloss',not any(c['rule']=='unit_name_gloss' for c in e.candidates('The 1s orbital was occupied.',SourceIndex('The 1s orbital was occupied.'))))
for text in ['The angular momentum was 3 h in this notation.','The gravitational acceleration was 5 g during the test.']:
 check('ambiguous symbol excluded '+text,not any(c['rule']=='unit_name_gloss' for c in e.candidates(text,SourceIndex(text))))
text='The sample was $m = 5 mg$ in this notation.'
check('formula interior protected',not any(15<c['start']<22 for c in e.candidates(text,SourceIndex(text))))
text='The system was held at 400 K and 1000 atm during evaluation.'
c=e.candidates(text,SourceIndex(text));check('mixed-dimension prefix protected',not any(c['rule'] in {'typed_temperature','explicit_temperature'} for c in c))
text='The signal was measured with high accuracy using software.'
check('accuracy not instrument',not any(c['rule']=='instrument_with' for c in e.candidates(text,SourceIndex(text))))
# Both available insertions are individually zero-gain. Their combination breaks
# the middle five original words out of a20-word copied chain.
text='The sample remained stable for hours at 25 °C while monitored at 30 °C throughout the duration of the experiment.'
index=SourceIndex(text)
a,_,_=e.repair(text,index,'system_greedy');b,traces,stats=e.repair(text,index,'system_joint')
check('zero-gain greedy stalls',a==text)
check('joint plan makes real progress',b!=text and traces[0]['metrics_after']['original_covered']<len(tokenize(text)[1]))
check('zero-gain intermediate admitted',stats['zero_gain_intermediate_plans']>=1)
check('joint provenance',e.verify_trace(text,b,traces))
for text in ['The sample contained 5 mg of the tested compound.','The results shown in Table 1 support the hypothesis.']:
 after,traces,_=e.repair(text,SourceIndex(text),'system_joint')
 check('real edit/provenance '+text,after!=text and e.verify_trace(text,after,traces))
check('reject quantity change',not e.scientific_guard('The mass was 5 mg.','The mass was 50 mg.'))
check('reject changed modality',not e.scientific_guard('The method may work.','The method works.'))
check('reject extra certainty word',not e.scientific_guard('The method works.','The method clearly works.'))
text='The sample contained 5 g of the tested compound.'
after,traces,_=e.repair(text,SourceIndex(text),'system_joint')
check('single-letter unit typed gloss',after!=text and '(grams)' in after and e.verify_trace(text,after,traces,SourceIndex(text)))
check('reject added certainty',all('clearly' not in c['replacement'] and 'maybe' not in c['replacement'] for c in e.candidates('The method was tested at 25 °C.',SourceIndex('The method was tested at 25 °C.'))))
print('PASS',len(checks),'catalog/provenance/joint-planning checks')
print('JOINT_FIXTURE',b)
print('JOINT_FIXTURE_STATS',stats)
