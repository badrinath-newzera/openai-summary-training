import functools
from lm_human_preferences.train_reward import HParams, train, RunHParams, LabelHParams
from lm_human_preferences.lm_tasks import TaskHParams, PolicyHParams


call = functools.partial(train, HParams(run=RunHParams(seed=1, log_interval=10, save_interval=50, save_dir='/home/badri/mansion/openai-training/lm-human-preferences/tmp/save/train_reward/testdesc-2106171021'), task=TaskHParams(query_length=64, query_dataset='books', query_prefix='', query_suffix='', start_text='.', end_text='.', response_length=24, truncate_token=13,
                                                                                                                                                                                                                                   truncate_after=16, penalty_reward_value=-1, policy=PolicyHParams(temperature=0.7, initial_model='124M')), labels=LabelHParams(type='compare', num_train=4992, source='https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json'), batch_size=32, lr=5e-05, rollout_batch_size=512, normalize_samples=256, debug_normalize=0, normalize_before=True, normalize_after=True))


call()
