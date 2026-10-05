import unittest
from passages import patch, assemble, select
from ngram_overlap import fragments

class PassageTests(unittest.TestCase):
    def test_identity_and_only_narrative_replaced(self):
        summary={'title':'Paper A', 'executive_summary':'First finding. Second finding.',
                 'limitations':[{'This is a limitation.':'exact quote'}]}
        self.assertEqual(patch(summary,{}),summary)
        original=[v for v in fragments(summary) if v[0]=='This is a limitation.'][0]
        out=patch(summary,{(original[1],original[0]):'A restriction applies.'})
        self.assertEqual(out['title'],summary['title'])
        self.assertEqual(out['executive_summary'],summary['executive_summary'])
        self.assertEqual(out['limitations'],[{'A restriction applies.':'exact quote'}])
    def test_window_offsets_are_disjoint_and_preserve_nonprose(self):
        text='Our first statement has nine matching words from this source. Another sentence describes results.'
        summary={'title':'T', 'executive_summary':text}
        wins=select(summary,text)
        self.assertTrue(wins)
        for left,right in zip(wins,wins[1:]): self.assertLessEqual(left['end'],right['start'])
        result=assemble(summary,wins,['A revised sentence.' for w in wins])
        self.assertEqual(result['title'],'T')
        self.assertIn('A revised sentence.',result['executive_summary'])

if __name__=='__main__':unittest.main()
