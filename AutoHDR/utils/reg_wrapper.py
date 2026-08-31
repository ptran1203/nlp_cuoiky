import subprocess
import os
import torch
import pickle
import atexit
import sys

from utils.wsl_bridge import is_windows, wsl_command

class reg_model:
    def __init__(self, executable_path: str = './dist/model_exe'):
        self.executable_path = executable_path
        self.process = None

    def start(self):
        """启动进程"""
        if self.process is None or self.process.poll() is not None:
            print('Starting process...', file=sys.stderr, flush=True)
            # dist/model_exe is a Linux ELF executable - no Windows build exists, so on Windows
            # this launches it inside WSL2 instead (see utils/wsl_bridge.py). `wsl.exe` proxies
            # stdin/stdout/stderr transparently, so __call__'s pipe protocol below is unchanged.
            cmd = wsl_command(self.executable_path) if is_windows() else [self.executable_path]

            # NOT hiding the GPU here, unlike det_wrapper.py. Confirmed by actually running it:
            # reg_model's checkpoint was saved with CUDA-tagged tensors, and its own torch.load()
            # call (inside the compiled binary, can't patch it) doesn't pass map_location='cpu' -
            # with CUDA_VISIBLE_DEVICES='' hiding the GPU, deserialization itself fails outright:
            # "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is
            # False." det_model's checkpoint loads fine either way (confirmed separately), so only
            # this wrapper needs the GPU left visible - on a machine with a real, working GPU
            # (Colab), that's not a problem; __call__ below defaults to 'cuda' accordingly.
            env = os.environ.copy()

            # stderr redirected to a log FILE, not PIPE - reading a PIPE only reliably works once
            # the process has actually exited, and stdout must stay a PIPE (it carries the real
            # length-prefixed protocol), so there's no clean way to also read stderr on failure
            # without either a background thread or a file. A file also survives even if
            # cleanup() runs before we get to read it. This is exactly what surfaced the
            # CUDA-deserialization error above - without it the failure was silent.
            self._log_path = os.path.abspath('reg_model_subprocess.log')
            self._log_file = open(self._log_path, 'w')
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_file,
            )
            print(f'Process started with PID {self.process.pid}', file=sys.stderr, flush=True)

    def _read_log_tail(self, max_chars: int = 4000) -> str:
        try:
            with open(self._log_path, 'r', errors='replace') as f:
                text = f.read()
            return text[-max_chars:] if len(text) > max_chars else text
        except Exception as e:
            return f"(couldn't read log: {e})"

    def __call__(self, x: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Process is not running")

        try:
            input_data = {
                'tensor': x.to(device),
                'device': device
            }

            print('Serializing input data...', file=sys.stderr, flush=True)
            data = pickle.dumps(input_data)

            # 先发送数据长度
            print('Sending data length...', file=sys.stderr, flush=True)
            self.process.stdin.write(len(data).to_bytes(4, byteorder='big'))

            # 发送数据
            print('Sending data...', file=sys.stderr, flush=True)
            self.process.stdin.write(data)
            self.process.stdin.flush()

            # 读取响应长度
            print('Reading response length...', file=sys.stderr, flush=True)
            length_bytes = self.process.stdout.read(4)
            if not length_bytes:
                raise EOFError("Process terminated unexpectedly")
            length = int.from_bytes(length_bytes, byteorder='big')

            # 读取响应数据
            print(f'Reading response data ({length} bytes)...', file=sys.stderr, flush=True)
            output_data = self.process.stdout.read(length)
            output = pickle.loads(output_data)
            print('Response received', file=sys.stderr, flush=True)

            return output['tensor']

        except Exception as e:
            print(f"Error during communication: {e}", file=sys.stderr, flush=True)
            if getattr(self, '_log_file', None) is not None:
                self._log_file.flush()
            print(f"Process poll: {self.process.poll()}", file=sys.stderr, flush=True)
            print(f"Log tail ({getattr(self, '_log_path', '?')}):", file=sys.stderr, flush=True)
            print(self._read_log_tail(), file=sys.stderr, flush=True)
            raise

    def cleanup(self):
        """清理资源"""
        if self.process is not None:
            print('Cleaning up process...', file=sys.stderr, flush=True)
            if self.process.poll() is None:  # 如果进程还在运行
                try:
                    self.process.stdin.close()
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    print('Process not responding, terminating...',
                          file=sys.stderr, flush=True)
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        print('Process still not responding, killing...',
                              file=sys.stderr, flush=True)
                        self.process.kill()
                        self.process.wait()

            self.process.stdout.close()
            if getattr(self, '_log_file', None) is not None:
                try:
                    self._log_file.close()
                except:
                    pass
                self._log_file = None
            self.process = None
            print('Process cleaned up', file=sys.stderr, flush=True)

    def __del__(self):
        """析构函数"""
        self.cleanup()
