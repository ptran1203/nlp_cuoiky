import subprocess
import torch
import pickle
import sys,os
import time
import socket
from typing import Optional

from utils.wsl_bridge import is_windows, wsl_command

class det_model:
    def __init__(self, executable_path: str = './dist/det_model/det_model', port: int = 12345, max_retries: int = 60):
        self.executable_path = executable_path
        self.process = None
        self.stride = None
        self._is_running = False
        self.port = port
        self.sock = None
        # Bumped from the original 5 (~10s budget): launched via WSL2 on Windows, cold start
        # (WSL2 process start + full torch import inside the frozen binary) takes ~20-25s before
        # the socket is actually listening - see utils/wsl_bridge.py.
        self.max_retries = max_retries

    def _read_log_tail(self, max_chars: int = 4000) -> str:
        try:
            with open(self._log_path, 'r', errors='replace') as f:
                text = f.read()
            return text[-max_chars:] if len(text) > max_chars else text
        except Exception as e:
            return f"(couldn't read log: {e})"

    def _connect_with_retry(self):
        """尝试连接服务器，带重试机制"""
        retries = 0
        while retries < self.max_retries:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(('localhost', self.port))
                return True
            except ConnectionRefusedError:
                print(f"Connection attempt {retries + 1} failed, retrying...", file=sys.stderr)
                retries += 1
                time.sleep(2)  # 等待2秒后重试
                continue
        return False

    def start(self):
        try:
            executable_path = os.path.abspath(self.executable_path)
            print(f"Starting process: {executable_path}")

            # dist/det_model/det_model is a Linux ELF executable - no Windows build exists, so
            # on Windows this launches it inside WSL2 instead (see utils/wsl_bridge.py). The
            # original hardcoded relative path ('dist/det_model/det_model') is kept as-is for
            # the non-Windows case to match upstream behavior exactly.
            if is_windows():
                cmd = wsl_command('dist/det_model/det_model', str(self.port))
            else:
                cmd = ['dist/det_model/det_model', str(self.port)]

            # Force this subprocess to see no GPU at all, regardless of platform. Confirmed on
            # WSL2 (no GPU visible there at all) that the binary starts reliably in ~20-25s and
            # falls back to CPU cleanly. On a machine where a real GPU IS visible (e.g. Colab),
            # this same binary hung/never bound its socket ("Failed to connect to server after
            # multiple attempts" even with a 120s retry budget) - the frozen binary bundles its
            # own CUDA runtime at build time, and a version/driver mismatch against whatever GPU
            # is actually present can hang CUDA init before the process ever gets to opening the
            # socket. Hiding the GPU here reproduces the one config that's actually been proven
            # to work, and the __call__ device-forcing below already assumes CPU-only regardless.
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''

            # Also capture stdout/stderr to a log file instead of just inheriting the parent's -
            # this process prints a LOT (PyInstaller loader + Python import tracing), enough to
            # bury the one line that actually matters if something goes wrong again.
            self._log_path = os.path.abspath('det_model_subprocess.log')
            self._log_file = open(self._log_path, 'w')
            self.process = subprocess.Popen(cmd, env=env, stdout=self._log_file, stderr=subprocess.STDOUT)
            # 检查进程是否立即退出
            time.sleep(1)
            exit_code = self.process.poll()
            if exit_code is not None:
                self._log_file.flush()
                print(f"Process exited immediately with code: {exit_code}")
                print(f"Log tail ({self._log_path}):")
                print(self._read_log_tail())
                raise RuntimeError(f"Process exited with code {exit_code}")

            print("Process started successfully")

            # 等待服务器启动并尝试连接
            if not self._connect_with_retry():
                self._log_file.flush()
                print(f"Log tail ({self._log_path}) - process was still running but never bound its socket:")
                print(self._read_log_tail())
                raise RuntimeError("Failed to connect to server after multiple attempts")
            
            print(f"Process started with PID: {self.process.pid}")
            self._is_running = True
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to start process: {str(e)}")

    def _send_data(self, data):
        """发送数据到服务器"""
        serialized_data = pickle.dumps(data)
        length = len(serialized_data)
        self.sock.sendall(length.to_bytes(4, byteorder='big'))
        self.sock.sendall(serialized_data)

    def _recv_data(self):
        """从服务器接收数据"""
        length_bytes = self.sock.recv(4)
        if not length_bytes:
            raise RuntimeError("Connection closed by server")
        length = int.from_bytes(length_bytes, byteorder='big')
        
        data = b''
        while len(data) < length:
            chunk = self.sock.recv(length - len(data))
            if not chunk:
                raise RuntimeError("Connection closed by server")
            data += chunk
            
        result = pickle.loads(data)
        if 'error' in result:
            raise RuntimeError(f"Server error: {result['error']}")
        return result

    def __call__(self, x, mode: int = 2):
        if not self._is_running:
            self.start()

        # The subprocess always runs with CUDA_VISIBLE_DEVICES='' now (see start()) - a CUDA
        # tensor or device object pickled here would fail to unpickle there
        # (torch.cuda.is_available() is False in that process), so force CPU regardless of what
        # device the caller (infer_pipeline.py) intended for its own (GPU) process. Also force
        # float32: infer_pipeline.py casts to .half() for its own CUDA path, and .cpu() alone
        # doesn't change dtype - a float16 tensor stays float16, but the subprocess's own model
        # is float32 (CPU + half precision is generally unsupported/pointless), producing
        # "Input type (c10::Half) and bias type (float) should be the same".
        if isinstance(x, torch.Tensor):
            x = x.cpu().float()
        elif isinstance(x, torch.device):
            x = torch.device('cpu')

        try:
            if mode == 1:
                print(f"mode 1 begin")
                print(f"x: {x}, mode: {mode}")
                
                # 发送数据
                self._send_data({'x': x, 'mode': mode})
                
                # 接收响应
                output = self._recv_data()
                self.stride = output['stride']
                return self.stride
                
            elif mode == 2:
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"For mode 2, input should be torch.Tensor, got {type(x)}")

                # Diagnostic: confirm from the CLIENT side (not trusting the server's swallowed
                # traceback) that the cpu().float() cast above actually took effect on the real
                # tensor being sent, not just in theory.
                print(f"mode 2: sending tensor dtype={x.dtype}, shape={tuple(x.shape)}, device={x.device}")

                # 发送数据
                self._send_data({'x': x, 'mode': mode})
                
                # 接收响应
                output = self._recv_data()
                return output['tensor']
            else:
                raise ValueError(f"Invalid mode: {mode}")

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error during operation: {str(e)}")

    def cleanup(self):
        """清理资源"""
        if self.sock is not None:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            
        if getattr(self, '_log_file', None) is not None:
            try:
                self._log_file.close()
            except:
                pass
            self._log_file = None

        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except:
                self.process.kill()
            finally:
                self.process = None
                self._is_running = False

    def __del__(self):
        self.cleanup()

if __name__ == '__main__':
    # 测试代码
    try:
        model = det_model('dist/det_model/det_model')
        model.start()
        print("Process started successfully")
        
        # 测试初始化
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        stride = model(device, mode=1)
        print(f"Model initialized with stride: {stride}")
        
        # 测试推理
        img = torch.randn(1, 3, 640, 640)
        output = model(img, mode=2)
        print(f"Inference successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)