from __future__ import annotations

import base64
import builtins as _builtins
import io
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict, List
import time

import pandas as pd
import numpy as np
import matplotlib

# Use a non-interactive backend for headless figure rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # after backend selection

import multiprocessing as mp
import warnings
import platform


_ALLOWED_TOPLEVEL_IMPORTS = {
    "pandas",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "sklearn",
    "sklearn.model_selection",
    "sklearn.linear_model",
    "sklearn.preprocessing",
    "sklearn.metrics",
    "sklearn.tree",
    "warnings",
    "tabulate",
    "streamlit",
    "psutil",
    "os",
}


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
    top = name.split(".", 1)[0]
    if name not in _ALLOWED_TOPLEVEL_IMPORTS and top not in {m.split(".", 1)[0] for m in _ALLOWED_TOPLEVEL_IMPORTS}:
        raise ImportError(f"Import of '{name}' is not allowed")
    return _builtins.__import__(name, globals, locals, fromlist, level)


def _make_restricted_builtins() -> Dict[str, Any]:
    safe = dict(_builtins.__dict__)
    # Override importer with our restricted version
    safe["__import__"] = _restricted_import
    # Optionally remove obviously dangerous builtins
    for k in ("open", "eval", "exec", "compile", "input", "help", "__loader__", "__spec__"):
        if k in safe:
            safe.pop(k)
    return safe


def _encode_figures(fig_nums: List[int]) -> List[str]:
    images_b64: List[str] = []
    print(f"DEBUG: _encode_figures called with fig_nums: {fig_nums}")
    for num in fig_nums:
        try:
            print(f"DEBUG: Trying to get figure {num}")
            fig = plt.figure(num)
            print(f"DEBUG: Successfully got figure {num}")
        except Exception as e:
            print(f"DEBUG: Failed to get figure {num}: {e}")
            continue
        buf = io.BytesIO()
        try:
            print(f"DEBUG: Saving figure {num} to buffer")
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            images_b64.append(base64.b64encode(buf.read()).decode("ascii"))
            print(f"DEBUG: Successfully encoded figure {num}")
        except Exception as e:
            print(f"DEBUG: Failed to save/encode figure {num}: {e}")
        finally:
            buf.close()
            plt.close(fig)
    print(f"DEBUG: _encode_figures returning {len(images_b64)} images")
    return images_b64


def _worker_entry(code: str, df: pd.DataFrame, q: mp.Queue) -> None:
    stdout_capture = io.StringIO()
    error_msg: str | None = None
    figures_b64: List[str] = []

    restricted_builtins = _make_restricted_builtins()
    
    # Create a simple mock for st.dataframe
    class MockStreamlit:
        def __init__(self, table_titles):
            self.dataframes = []
            self.dataframe_count = 0
            self.table_titles = table_titles
            self.title_index = 0
        
        def dataframe(self, data, **kwargs):
            # No limits on dataframes - capture all for display
            self.dataframe_count += 1
            
            # Get the title for this dataframe call
            title = None
            if self.title_index < len(self.table_titles):
                title = self.table_titles[self.title_index]
                self.title_index += 1
            
            # Capture the dataframe call for later display with title
            self.dataframes.append({
                'data': data,
                'kwargs': kwargs,
                'title': title,
                'call_order': self.dataframe_count  # Track the order of calls
            })
            print(f"DEBUG: Captured dataframe {self.dataframe_count} with title: {title}")
            # Don't print anything - let the captured DataFrames be displayed by the main app
        
        def write(self, *args, **kwargs):
            # Mock st.write() - just print to stdout for now
            # This will be captured by the stdout capture
            print(*args)
        
        def selectbox(self, label, options, **kwargs):
            return options[0] if options else None
    
    # No figure limits - allow unlimited plots
    print("DEBUG: No figure limits applied - unlimited plots allowed")
    
    # Parse the code to extract table titles from comments
    lines = code.split('\n')
    table_titles = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('# Table:'):
            # Extract the title from the comment
            title = line.replace('# Table:', '').strip()
            table_titles.append(title)
            print(f"DEBUG: Found table title: {title}")
    
    print(f"DEBUG: Total table titles found: {len(table_titles)}")
    
    mock_st = MockStreamlit(table_titles)
    
    exec_globals: Dict[str, Any] = {
        "__builtins__": restricted_builtins,  # Use regular dict instead of MappingProxyType
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "st": mock_st,
    }
    
    # Add seaborn with some safety settings
    try:
        import seaborn as sns
        # Set seaborn to use non-interactive backend
        sns.set_theme(style="whitegrid", context="notebook")
        exec_globals["sns"] = sns
        print("DEBUG: Seaborn imported successfully")
        print(f"DEBUG: Seaborn style set to: {sns.axes_style()}")
    except Exception as e:
        print(f"DEBUG: Seaborn import failed: {e}")
        exec_globals["sns"] = None
    
    # Test sklearn imports
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        exec_globals.update({
            'train_test_split': train_test_split,
            'StandardScaler': StandardScaler,
            'LogisticRegression': LogisticRegression,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'accuracy_score': accuracy_score,
            'roc_auc_score': roc_auc_score
        })
        print("DEBUG: Sklearn imports successful")
    except Exception as e:
        print(f"DEBUG: Sklearn imports failed: {e}")
        exec_globals.update({
            'train_test_split': None,
            'StandardScaler': None,
            'LogisticRegression': None,
            'DecisionTreeClassifier': None,
            'accuracy_score': None,
            'roc_auc_score': None
        })
    # Suppress noisy non-interactive backend warnings from plt.show()
    warnings.filterwarnings(
        "ignore",
        message=".*FigureCanvasAgg is non-interactive, and thus cannot be shown.*",
        category=UserWarning,
    )
    
    # Send initial status
    print(f"DEBUG: Worker sending 'started' message")
    q.put({"status": "started"})
    
    try:
        print(f"DEBUG: Worker starting code execution")
        print(f"DEBUG: DataFrame info - shape: {df.shape}, columns: {list(df.columns)}")
        print(f"DEBUG: DataFrame column types: {[type(col) for col in df.columns]}")
        print(f"DEBUG: First few rows of DataFrame:")
        print(df.head(2))
        
        # Skip data type conversion to avoid hanging - let LLM code handle it
        print(f"DEBUG: Skipping data type conversion to avoid hanging - LLM code will handle data types")
        print(f"DEBUG: DataFrame dtypes: {df.dtypes.to_dict()}")
        print(f"DEBUG: Using full dataset with {len(df)} rows for complete analysis")
        
        # Set matplotlib to non-interactive mode to prevent hanging
        print("DEBUG: Setting up matplotlib...")
        plt.ioff()
        # Set backend to Agg for better performance
        plt.switch_backend('Agg')
        print(f"DEBUG: Matplotlib backend set to: {plt.get_backend()}")
        
        # Test figure creation
        print("DEBUG: Creating test figure...")
        test_fig = plt.figure()
        print(f"DEBUG: Test figure created: {test_fig.number}")
        
        # Test a simple plot
        print("DEBUG: Adding test plot...")
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Test Plot")
        print(f"DEBUG: Test plot added to figure {test_fig.number}")
        
        plt.close(test_fig)
        print("DEBUG: Test figure closed successfully")
        
        # Add progress messages to identify where it hangs
        print("DEBUG: About to execute code...")
        print("DEBUG: Starting exec() call...")
        print(f"DEBUG: Code length: {len(code)} characters")
        print(f"DEBUG: Exec globals keys: {list(exec_globals.keys())}")
        
        # Add system information
        import psutil
        import os
        print(f"DEBUG: Process PID: {os.getpid()}")
        print(f"DEBUG: Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"DEBUG: CPU count: {psutil.cpu_count()}")
        print(f"DEBUG: Available memory: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB")
        
        # Try to execute the code with more granular debugging
        try:
            with redirect_stdout(stdout_capture):
                print("DEBUG: Inside redirect_stdout context")
                print("DEBUG: About to call exec()")
                exec(code, exec_globals, None)
                print("DEBUG: exec() call returned successfully")
        except Exception as e:
            print(f"DEBUG: exec() call failed with exception: {e}")
            raise
        
        print("DEBUG: exec() call completed successfully")
        print(f"DEBUG: After execution, matplotlib has {len(plt.get_fignums())} figures: {plt.get_fignums()}")
        print("DEBUG: Code execution completed successfully")
        # Send completion status
        print(f"DEBUG: Worker sending 'completed' message")
        q.put({"status": "completed"})
    except KeyError as e:
        error_msg = f"Column not found: {str(e)}. Available columns: {list(df.columns)}"
        print(f"DEBUG: Worker sending 'error' message: {error_msg[:100]}...")
        print(f"DEBUG: Column names with repr: {[repr(col) for col in df.columns]}")
        print(f"DEBUG: Error details: {repr(str(e))}")
        q.put({"status": "error", "error": error_msg})
    except NameError as e:
        error_msg = f"Variable not defined: {str(e)}. Available variables: df, pd, np, plt, sns, sklearn, tabulate, st"
        print(f"DEBUG: Worker sending 'error' message: {error_msg[:100]}...")
        q.put({"status": "error", "error": error_msg})
    except ImportError as e:
        error_msg = f"Import error: {str(e)}. This might be due to restricted imports."
        print(f"DEBUG: Worker sending 'error' message: {error_msg[:100]}...")
        q.put({"status": "error", "error": error_msg})
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"DEBUG: Worker sending 'error' message: {error_msg[:100]}...")
        print(f"DEBUG: FULL ERROR TRACEBACK:")
        print(error_msg)
        q.put({"status": "error", "error": error_msg})
    finally:
        print(f"DEBUG: Worker in finally block, capturing figures")
        # Always try to capture figures and stdout, even on error/timeout
        try:
            # Check if matplotlib is still available
            print(f"DEBUG: Checking matplotlib state...")
            print(f"DEBUG: plt.get_fignums() available: {hasattr(plt, 'get_fignums')}")
            
            fig_nums = sorted(list(plt.get_fignums()))
            print(f"DEBUG: Found figure numbers: {fig_nums}")
            print(f"DEBUG: Number of figures found: {len(fig_nums)}")
            
            if not fig_nums:
                print("DEBUG: No figures found! Checking if matplotlib is working...")
                # Try to create a test figure to see if matplotlib is working
                try:
                    test_fig = plt.figure()
                    print(f"DEBUG: Test figure created successfully: {test_fig.number}")
                    fig_nums = [test_fig.number]
                    plt.close(test_fig)
                except Exception as test_e:
                    print(f"DEBUG: Test figure creation failed: {test_e}")
            
            figures_b64 = _encode_figures(fig_nums)
            print(f"DEBUG: Worker captured {len(figures_b64)} figures")
        except Exception as e:
            print(f"DEBUG: Worker figure capture failed: {e}")
            print(f"DEBUG: Full figure capture error: {traceback.format_exc()}")
            figures_b64 = []
        
        # Send final results
        # Clean stdout to remove debug messages and section markers
        stdout_content = stdout_capture.getvalue()
        
        # Remove debug messages that might have been captured
        stdout_lines = stdout_content.split('\n')
        cleaned_lines = []
        for line in stdout_lines:
            # Skip debug messages
            if line.startswith('DEBUG:') or 'DEBUG:' in line:
                continue
            # Skip matplotlib figure info
            if 'matplotlib has' in line and 'figures:' in line:
                continue
            if 'exec() call completed' in line:
                continue
            if 'Code execution completed' in line:
                continue
            if 'Keeping figures open' in line:
                continue
            cleaned_lines.append(line)
        
        cleaned_stdout = '\n'.join(cleaned_lines)
        
        # Remove markdown code blocks from stdout
        import re
        cleaned_stdout = re.sub(r'```[a-zA-Z]*\n', '', cleaned_stdout)  # Remove ```python, ```, etc.
        cleaned_stdout = re.sub(r'\n```\n', '\n', cleaned_stdout)  # Remove closing ```
        cleaned_stdout = re.sub(r'^```\n', '', cleaned_stdout)  # Remove opening ``` at start
        cleaned_stdout = re.sub(r'\n```$', '', cleaned_stdout)  # Remove closing ``` at end
        
        final_payload = {
            "status": "final",
            "stdout": cleaned_stdout,
            "figures": figures_b64,
            "dataframes": mock_st.dataframes,  # Include captured dataframes
            "error": error_msg,
        }
        print(f"DEBUG: Worker sending 'final' message with {len(final_payload['stdout'])} chars stdout, {len(figures_b64)} figures, {len(mock_st.dataframes)} dataframes")
        q.put(final_payload)
        # Wait a moment to ensure the message is sent
        time.sleep(0.1)
        print(f"DEBUG: Worker completed")


def _get_ctx() -> mp.context.BaseContext:
    # On macOS, prefer 'spawn' to avoid potential hanging issues with 'fork'
    if platform.system() == "Darwin":  # macOS
        try:
            return mp.get_context("spawn")
        except Exception:
            pass
    # Try 'fork' first, fallback to 'spawn' if not available
    try:
        return mp.get_context("fork")
    except Exception:
        pass
    try:
        return mp.get_context("spawn")
    except Exception:
        pass
    return mp.get_context("fork")


def execute_code(code: str, df: pd.DataFrame, timeout_seconds: int = 30) -> Dict[str, Any]:
    """Execute user EDA code safely and return stdout, figures, error.

    - Runs inside a restricted namespace with only df, pd, np, plt.
    - Allows imports only from pandas/numpy/matplotlib/seaborn/sklearn.
    - Captures stdout and any matplotlib figures as base64-encoded PNGs.
    - Enforces a timeout via a separate process (works outside main thread).
    - Collects partial results even on timeout.
    """
    ctx = _get_ctx()
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker_entry, args=(code, df, q))
    proc.start()
    
    result: Dict[str, Any] = {
        "stdout": "",
        "figures": [],
        "error": None,
    }
    
    # Monitor the process and collect results
    start_time = time.time()
    timeout_reached = False
    final_results_received = False
    
    print(f"DEBUG: Starting execution monitoring with timeout {timeout_seconds}s")
    print(f"DEBUG: Process started with PID: {proc.pid}")
    
    while proc.is_alive() and (time.time() - start_time) < timeout_seconds:
        try:
            # Non-blocking check for messages
            payload = q.get_nowait()
            print(f"DEBUG: Received message: {payload.get('status', 'unknown')}")
            if isinstance(payload, dict):
                if payload.get("status") == "final":
                    # Final results received
                    print(f"DEBUG: Received final results, breaking")
                    result.update(payload)
                    final_results_received = True
                    break
                elif payload.get("status") == "error":
                    # Error occurred
                    print(f"DEBUG: Received error message")
                    result["error"] = payload.get("error")
                    final_results_received = True
                    break
                # Other status messages (started, completed) are just for monitoring
        except Exception as e:
            # No message available, continue monitoring
            pass
        time.sleep(0.1)  # Small delay to avoid busy waiting
    
    # Only terminate if we actually hit timeout (not if we got final results)
    if proc.is_alive() and not final_results_received:
        timeout_reached = True
        print(f"DEBUG: Process still alive after {time.time() - start_time:.1f}s, terminating")
        print(f"DEBUG: Process still alive: {proc.is_alive()}")
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            try:
                proc.kill()
                print(f"DEBUG: Process killed due to timeout")
            except Exception:
                pass
            finally:
                proc.join(1)
    else:
        print(f"DEBUG: Process completed successfully in {time.time() - start_time:.1f}s")
    
    # Try to collect any remaining results from queue (only if we hit timeout)
    if timeout_reached:
        print(f"DEBUG: Timeout reached, trying to collect remaining results")
        try:
            while True:
                payload = q.get_nowait()
                print(f"DEBUG: Collected remaining message: {payload.get('status', 'unknown')}")
                if isinstance(payload, dict) and payload.get("status") == "final":
                    result.update(payload)
                    print(f"DEBUG: Updated result with final payload")
                    break
        except Exception as e:
            print(f"DEBUG: No more messages in queue: {e}")
            pass
        
        # Add timeout message to error
        if result["error"]:
            result["error"] = f"Execution exceeded {timeout_seconds}s timeout\n\n{result['error']}"
        else:
            result["error"] = f"Execution exceeded {timeout_seconds}s timeout"
    
    # If we still don't have results, try one more time
    if not result.get("stdout") and not result.get("figures") and not result.get("error"):
        print(f"DEBUG: No results yet, trying one final queue check")
        try:
            # Try multiple times to get the final message
            for _ in range(5):
                try:
                    payload = q.get_nowait()
                    print(f"DEBUG: Final queue check got: {payload.get('status', 'unknown')}")
                    if isinstance(payload, dict):
                        result.update(payload)
                        if payload.get("status") == "final":
                            break
                except Exception:
                    time.sleep(0.1)
                    continue
        except Exception as e:
            print(f"DEBUG: Final queue check failed: {e}")
            result["error"] = result["error"] or "No result returned from execution process"
    
    print(f"DEBUG: Execution completed. stdout: {len(result.get('stdout', ''))}, figures: {len(result.get('figures', []))}, error: {result.get('error')}")
    return result 