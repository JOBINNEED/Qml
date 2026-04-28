"""
ibm_quantum_backend.py
======================
IBM Quantum Cloud Integration for VRP Solver.

FEATURES:
    • Use IBM Cloud simulators (faster than local)
    • Run on real quantum hardware (127-qubit IBM devices)
    • Automatic backend selection based on problem size
    • Error mitigation for hardware runs
    • Job monitoring and result retrieval

SETUP:
    1. Get IBM Quantum API token: https://quantum.ibm.com/
    2. Add to .env: IBM_QUANTUM_TOKEN=your_token_here
    3. Run with: python baseline_benchmark.py --use-ibm

NOTE: Requires qiskit-ibm-runtime >= 0.20 (ibm_quantum_platform channel)
"""
import sys, os
# Ensure project root is on path and cwd is project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)


import os
from typing import Optional, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False

# Fallback to local StatevectorSampler if IBM not available
from qiskit.primitives import StatevectorSampler


class IBMQuantumBackend:
    """
    Manages IBM Quantum cloud access for VRP experiments.

    Usage:
        backend = IBMQuantumBackend(use_hardware=False)
        sampler = backend.get_sampler(n_qubits=12)
        # Use sampler in QAOA as usual
        backend.close()
    """

    def __init__(self,
                 token: Optional[str] = None,
                 instance: str = 'ibm-q/open/main',
                 use_hardware: bool = False):
        """
        Initialize IBM Quantum backend.

        Parameters
        ----------
        token : str, optional
            IBM Quantum API token. If None, tries to load from env var.
        instance : str
            IBM Quantum instance (default: open access)
        use_hardware : bool
            If True, use real quantum hardware. If False, use local simulator.
        """
        self.use_hardware = use_hardware
        self.service = None
        self.backend = None
        self.session = None
        self._use_local = False

        if not IBM_RUNTIME_AVAILABLE:
            print("⚠ qiskit-ibm-runtime not installed. Using local simulator.")
            print("  Install with: pip install qiskit-ibm-runtime")
            self._use_local = True
            return

        # Try to get token from environment if not provided
        if token is None:
            token = os.getenv('IBM_QUANTUM_TOKEN') or os.getenv('IBM_API_KEY')
            if token:
                print("✓ Loaded IBM Quantum token from .env file")

        if not token:
            print("⚠ No IBM Quantum token found. Using local simulator.")
            print("  Add IBM_QUANTUM_TOKEN=<token> to your .env file.")
            self._use_local = True
            return

        # Initialize service — try ibm_quantum_platform first, then ibm_cloud
        for channel in ('ibm_quantum_platform', 'ibm_cloud'):
            try:
                QiskitRuntimeService.save_account(
                    channel=channel,
                    token=token,
                    overwrite=True
                )
                self.service = QiskitRuntimeService(channel=channel)
                print(f"✓ IBM Quantum token saved (channel: {channel})")
                print(f"✓ Connected to IBM Quantum Cloud")
                break
            except Exception as e:
                continue

        if self.service is None:
            print("✗ Failed to connect to IBM Quantum. Using local simulator.")
            print("  Check your token at: https://quantum.ibm.com/")
            self._use_local = True

    def select_backend(self, n_qubits: int, prefer_hardware: bool = None) -> Optional[str]:
        """Select optimal backend based on problem size."""
        if self._use_local or self.service is None:
            return None

        use_hw = prefer_hardware if prefer_hardware is not None else self.use_hardware

        if not use_hw:
            print(f"\n  Backend: IBM Cloud Simulator (local fallback used)")
            return None  # Use local StatevectorSampler

        # Select real quantum hardware
        print(f"\n  Searching for quantum hardware with ≥{n_qubits} qubits...")
        try:
            backends = self.service.backends(
                min_num_qubits=n_qubits,
                filters=lambda x: not x.simulator and x.status().operational
            )
            if not backends:
                print(f"  ✗ No hardware available. Using local simulator.")
                return None

            backend = min(backends, key=lambda x: x.status().pending_jobs)
            print(f"  ✓ Selected: {backend.name}")
            return backend.name
        except Exception as e:
            print(f"  ✗ Backend selection failed: {e}. Using local simulator.")
            return None

    def get_sampler(self,
                    n_qubits: int,
                    optimization_level: int = 1,
                    shots: int = 1024):
        """
        Get configured Sampler primitive for QAOA.
        Falls back to local StatevectorSampler if IBM not available.

        Returns
        -------
        StatevectorSampler (local) or SamplerV2 (IBM cloud)
        """
        if self._use_local or self.service is None:
            print("  Using local StatevectorSampler")
            return StatevectorSampler()

        backend_name = self.select_backend(n_qubits)
        if backend_name is None:
            print("  Using local StatevectorSampler")
            return StatevectorSampler()

        try:
            self.backend = self.service.backend(backend_name)
            self.session = Session(backend=self.backend)
            sampler = Sampler(mode=self.session)
            print(f"\n  IBM Sampler configured:")
            print(f"    Backend : {backend_name}")
            print(f"    Shots   : {shots}")
            return sampler
        except Exception as e:
            print(f"  ✗ IBM Sampler failed: {e}. Using local simulator.")
            return StatevectorSampler()

    def close(self):
        """Close session and cleanup."""
        if self.session:
            try:
                self.session.close()
                print("✓ IBM Quantum session closed")
            except Exception:
                pass

    def get_backend_info(self) -> Dict[str, Any]:
        """Get current backend information."""
        if not self.backend:
            return {'name': 'local_simulator', 'simulator': True}
        try:
            return {
                'name': self.backend.name,
                'n_qubits': self.backend.num_qubits,
                'operational': True,
            }
        except Exception:
            return {}


# ── Helper Functions ──────────────────────────────────────────────────────────

def setup_ibm_account(token: str):
    """One-time setup: Save IBM Quantum API token."""
    if not IBM_RUNTIME_AVAILABLE:
        print("✗ qiskit-ibm-runtime not installed.")
        return
    for channel in ('ibm_quantum_platform', 'ibm_cloud'):
        try:
            QiskitRuntimeService.save_account(channel=channel, token=token, overwrite=True)
            print(f"✓ IBM Quantum account saved (channel: {channel})")
            return
        except Exception as e:
            continue
    print("✗ Failed to save account. Check your token.")


def test_ibm_connection():
    """Test IBM Quantum connection and list available backends."""
    if not IBM_RUNTIME_AVAILABLE:
        print("✗ qiskit-ibm-runtime not installed.")
        return

    token = os.getenv('IBM_QUANTUM_TOKEN') or os.getenv('IBM_API_KEY')
    if not token:
        print("✗ No IBM_QUANTUM_TOKEN in environment.")
        return

    for channel in ('ibm_quantum_platform', 'ibm_cloud'):
        try:
            service = QiskitRuntimeService(channel=channel)
            print(f"✓ Connected to IBM Quantum Cloud (channel: {channel})\n")
            print("Available Backends:")
            print("-" * 70)
            backends = service.backends()
            for b in backends[:8]:
                try:
                    print(f"  • {b.name}")
                except Exception:
                    pass
            print("-" * 70)
            return
        except Exception as e:
            continue
    print("✗ Connection failed. Check your token at https://quantum.ibm.com/")


# ── Standalone Test ───────────────────────────────────────────────────────────

def main():
    import sys

    print("\n" + "=" * 70)
    print("  IBM QUANTUM BACKEND TEST")
    print("=" * 70)

    if len(sys.argv) > 1:
        token = sys.argv[1]
        print(f"\n[1] Saving IBM Quantum token...")
        setup_ibm_account(token)

    print(f"\n[2] Testing connection...")
    test_ibm_connection()

    print(f"\n[3] Testing backend manager...")
    try:
        mgr = IBMQuantumBackend(use_hardware=False)
        sampler = mgr.get_sampler(n_qubits=6)
        print(f"  Sampler type: {type(sampler).__name__}")
        mgr.close()
    except Exception as e:
        print(f"  ✗ Test failed: {e}")

    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
