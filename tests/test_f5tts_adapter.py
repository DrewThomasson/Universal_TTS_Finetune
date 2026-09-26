"""Mocked F5-TTS routing and control checks; these do not establish training or inference."""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('test_inference_dispatch_stubs', REPO/'tests/test_inference_dispatch.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)  # installs minimal heavy-dependency and Gradio stubs
pipeline = helper.pipeline
web_gui = helper.web_gui
from utils.model_registry import list_model_choices, pretrained_model_choices
from utils import f5tts_utils
from utils.e2a_export import export_e2a_zip

class F5MockedTests(unittest.TestCase):
    def test_registry_and_shared_language_mapping(self):
        self.assertIn(('F5-TTS v1 (English / Chinese)', 'f5_tts'), [(label,key) for key,label in list_model_choices()])
        for language in ('en','EN','zh','zh_CN','zh-cn'):
            self.assertTrue(pretrained_model_choices('f5_tts', language), language)
        for language in ('fr','zzzz'):
            self.assertFalse(pretrained_model_choices('f5_tts', language), language)

    def test_f5_inference_routes_to_optional_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); audio=root/'ok.wav'; audio.write_bytes(b'not actually audio, routing only'*3)
            artifacts={'family':'f5_tts','model_key':'f5_tts','artifacts_file':str(root/'artifacts.json'),'reference_wav':str(root/'ref.wav')}
            routed=Mock(return_value=audio)
            with patch.object(pipeline,'load_artifacts',return_value=artifacts), \
                 patch.dict(sys.modules, {'utils.f5tts_utils':types.SimpleNamespace(synthesize_f5tts=routed)}):
                result=pipeline.synthesize(artifacts_path_or_dir=folder,text='hello',language='en',speaker_wav=None,output_file=str(audio))
            routed.assert_called_once()
            self.assertEqual(result['model_key'],'f5_tts')
            self.assertEqual(result['output_file'],str(audio.resolve()))

    def test_train_validation_and_dry_run_delegate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            norm=patch.object(pipeline,'_normalize_dataset_dir',return_value=root)
            info=patch.object(pipeline,'load_dataset_info',return_value={})
            fake=Mock(return_value={'model_key':'f5_tts','status':'dry-run'})
            with norm, info, patch('utils.f5tts_utils.train_f5tts',fake):
                result=pipeline.train_model(model_key='f5_tts',output_root=folder,dataset_dir=folder,language='zh_CN',batch_size=1,dry_run=True)
            self.assertEqual(result['status'],'dry-run')
            self.assertEqual(fake.call_args.kwargs['language'],'zh-cn')
            self.assertEqual(fake.call_args.kwargs['batch_size'],1)
            for kwargs, message in [
                ({'language':'fr'},'supported starting checkpoint'),
                ({'language':'en','use_pretrained':False},'training from scratch'),
            ]:
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError,message):
                    pipeline.train_model(model_key='f5_tts',output_root=folder,dataset_dir=folder,**kwargs)
            with norm, info:
                for kwargs,message in [
                    ({'sample_epoch_interval':1},'periodic training samples'),
                    ({'pretrained_model_id':'wrong/base'},'Unsupported F5-TTS base checkpoint'),
                    ({'extra_overrides_json':'{"unknown": 1}'},'only the f5tts_python'),
                    ({'restore_path':'/tmp/local.pt'},'resume are not supported'),
                ]:
                    with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError,message):
                        pipeline.train_model(model_key='f5_tts',output_root=folder,dataset_dir=folder,language='en',batch_size=1,**kwargs)

    def test_gui_controls_clear_hidden_reference_and_share_choices(self):
        speaker, used=web_gui.on_model_change('f5_tts')
        self.assertEqual(speaker,{'visible':False,'value':None})
        self.assertEqual(used,{'visible':False,'value':None})
        self.assertEqual(web_gui.update_finetune_language_choices('f5_tts'),{'choices':['en','zh-cn'],'value':'en'})
        self.assertEqual(web_gui.get_adaptive_defaults('f5_tts',None),(10,1))
        resume=web_gui.update_resume_models('/tmp/unused','f5_tts')
        self.assertFalse(resume['interactive'])
        msg,_=web_gui.update_training_options('f5_tts','fr',True,None)
        self.assertIn('does not support',msg)

    def test_unsupported_language_and_custom_reference_rejected_without_runtime(self):
        with self.assertRaisesRegex(ValueError,'does not support language'):
            f5tts_utils.synthesize_f5tts({'reference_wav':'/tmp/ref.wav'},'hello','fr',None,'/tmp/out.wav')
        with self.assertRaisesRegex(ValueError,'different speaker WAV is unsupported'):
            f5tts_utils.synthesize_f5tts({'reference_wav':'/tmp/ref.wav'},'hello','zh_CN','/tmp/custom.wav','/tmp/out.wav')
        with self.assertRaisesRegex(ValueError,'local checkpoint overrides and resume'):
            f5tts_utils.train_f5tts(dataset_dir='.',training_root='.',language='en',epochs=2,batch_size=1,grad_accum=1,max_audio_seconds=11,restore_path='/tmp/local.pt',dry_run=True)

    def test_optional_worker_timeout_kills_process_and_removes_request(self):
        import signal
        from utils.f5tts_utils import _run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            proc=Mock(pid=123, returncode=None)
            proc.poll.return_value=None
            config={'python_executable':'/fake/f5-python'}
            with patch('utils.f5tts_utils.subprocess.Popen',return_value=proc), \
                 patch('utils.f5tts_utils.time.monotonic',side_effect=[0,5401]), \
                 patch('utils.f5tts_utils.os.killpg') as killpg:
                with self.assertRaisesRegex(TimeoutError,'exceeded its time limit'):
                    _run('train',config,root)
            killpg.assert_called_once_with(123,signal.SIGTERM)
            proc.wait.assert_called_once_with(timeout=10)
            self.assertEqual(list(root.glob('f5_train_*.json')),[])

    def test_e2a_export_rejects_f5(self):
        with self.assertRaisesRegex(ValueError,'does not support.*f5_tts'):
            export_e2a_zip({'model_key':'f5_tts'},'/tmp/no-f5-e2a.zip')

    def test_cli_model_choice_and_effective_default(self):
        import headless_cli
        args=headless_cli._build_parser().parse_args(['train','--model','f5_tts','--output-root','/tmp/out'])
        self.assertEqual(args.model,'f5_tts')
        self.assertIsNone(args.batch_size)
        # main() converts omitted F5 batch size to 1 for train as well as workflow.
        with patch.object(headless_cli,'train_model',return_value={'dry':'ok'}) as train, \
             patch.object(sys,'argv',['headless_cli.py','train','--model','f5_tts','--output-root','/tmp/out','--dry-run']):
            headless_cli.main()
        self.assertEqual(train.call_args.kwargs['batch_size'],1)

if __name__=='__main__':
    unittest.main(verbosity=2)
