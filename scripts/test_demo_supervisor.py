"""Exercise the actual supervisor with a harmless synthetic worker."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SupervisorTests(unittest.TestCase):
    def test_success_and_failure_produce_terminal_sidecars(self):
        source=Path(__file__).with_name('run_adaptive_demo.py').read_text()
        for code,expected in ((0,'finished'),(7,'failed')):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                base=Path(folder); script=base/'run_adaptive_demo.py'
                script.write_text(source)
                (base/'demo_adaptive_d.py').write_text(f'import sys; sys.exit({code})\n')
                out=base/'outputs'
                subprocess.run([sys.executable,'-B',str(script),'--root',str(out),
                    '--source-case',str(base),'--modes','reduced','--seconds','120'],
                    check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
                status=json.loads((out/'status.json').read_text())
                side=json.loads((out/'reduced-termination.json').read_text())
                self.assertTrue(status['finished'])
                self.assertEqual(side['state'],expected)
                self.assertEqual(side['exit_code'],code)
                self.assertFalse(side['result_exists'])


if __name__=='__main__': unittest.main()
