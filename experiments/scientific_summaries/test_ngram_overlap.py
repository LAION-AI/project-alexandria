import unittest
from ngram_overlap import SourceIndex,audit_summary


class OverlapTests(unittest.TestCase):
    def test_thresholds(self):
        source='one two three four five six seven eight'
        for count,expected in [(5,'within_5'),(6,'borderline_6'),(7,'violation_7plus')]:
            result=audit_summary(source,{'executive_summary':' '.join(source.split()[:count])})
            self.assertEqual(result['categories']['narrative']['classification'],expected)

    def test_no_cross_statement_false_match(self):
        result=audit_summary('one two three four five six seven',{
            'executive_summary':'one two three',
            'key_results':[{'four five six seven':'four five'}]})
        self.assertLessEqual(result['categories']['narrative']['longest_contiguous_match_words'],4)

    def test_punctuation_case_and_pdf_wrap(self):
        result=audit_summary('ONE, two three four five six seven inter-\nvention',
            {'executive_summary':'one two three four five six seven intervention'})
        self.assertEqual(result['categories']['narrative']['longest_contiguous_match_words'],8)

    def test_coverage_does_not_double_count_source_occurrences(self):
        phrase='one two three four five six seven eight'
        result=audit_summary(phrase+' filler '+phrase,{'executive_summary':phrase})
        self.assertEqual(result['categories']['narrative']['covered_words_by_minimum_run']['7'],8)

    def test_metadata_kept_separate(self):
        phrase='one two three four five six seven eight'
        result=audit_summary(phrase,{'title':phrase,'executive_summary':'A novel result.'})
        self.assertEqual(result['categories']['metadata']['longest_contiguous_match_words'],8)
        self.assertEqual(result['substantive_classification'],'within_5')

    def test_nested_raw_evidence_not_omitted(self):
        phrase='one two three four five six seven eight'
        result=audit_summary(phrase,{'key_results':[{'description':'Novel result','evidence':{'quote':phrase}}]})
        self.assertEqual(result['categories']['evidence']['longest_contiguous_match_words'],8)

    def test_long_copied_paragraph(self):
        phrase=' '.join('word'+str(i) for i in range(1000))
        result=audit_summary(phrase,{'executive_summary':phrase})
        self.assertEqual(result['categories']['narrative']['longest_contiguous_match_words'],1000)
        self.assertEqual(result['categories']['narrative']['covered_words_by_minimum_run']['7'],1000)

    def test_compatibility_accents_do_not_invent_extra_quote_words(self):
        phrase='−16 (ηkµ˜κνλ −ηµλ˜κνκ −ηνλ˜κµκ'
        result=audit_summary(phrase,{'key_results':[{'Paraphrased statement':phrase}]})
        self.assertEqual(result['categories']['evidence']['words'],4)
        self.assertLessEqual(result['categories']['evidence']['longest_contiguous_match_words'],5)


if __name__=='__main__':unittest.main()
