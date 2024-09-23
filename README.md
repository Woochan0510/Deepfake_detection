# Deepfake_detection
딥페이크가 사회적 이슈로 떠오름에 따른 딥페이크 탐지 프로그램
컴퓨터의 메모리 용량 부족으로 학습데이터가 부족함

2024-09-23 21:43:48.567475: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.       
Traceback (most recent call last):
  File "c:\Users\Asus\Desktop\coding\python\deepfake_detection\deepfake_detection.py", line 71, in <module>
    history = model.fit(
              ^^^^^^^^^^
  File "C:\Users\Asus\Desktop\coding\python\deepfake_detection\venv\Lib\site-packages\tensorflow\python\keras\engine\training.py", line 1137, in fit
    data_handler = data_adapter.get_data_handler(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Asus\Desktop\coding\python\deepfake_detection\venv\Lib\site-packages\tensorflow\python\keras\engine\data_adapter.py", line 1397, in get_data_handler
    return DataHandler(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Asus\Desktop\coding\python\deepfake_detection\venv\Lib\site-packages\tensorflow\python\keras\engine\data_adapter.py", line 1151, in __init__
    adapter_cls = select_data_adapter(x, y)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Asus\Desktop\coding\python\deepfake_detection\venv\Lib\site-packages\tensorflow\python\keras\engine\data_adapter.py", line 987, in select_data_adapter
    adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
                                                     ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Asus\Desktop\coding\python\deepfake_detection\venv\Lib\site-packages\tensorflow\python\keras\engine\data_adapter.py", line 706, in can_handle
    _is_distributed_dataset(x))
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Asus\Desktop\coding\python\deepfake_detection\venv\Lib\site-packages\tensorflow\python\keras\engine\data_adapter.py", line 1696, in _is_distributed_dataset
    return isinstance(ds, input_lib.DistributedDatasetInterface)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface'. Did you mean: 'DistributedDatasetSpec'?

위의 오류로 인한 실패

추후 더욱 공부하여 발전시킬것.
