import unittest
import torch

class TestCuda(unittest.TestCase):
    
    def test_cuda_availability(self):
        """Test if CUDA is available."""
        self.assertTrue(torch.cuda.is_available(), "CUDA is not available")
        
    def test_cuda_device_count(self):
        """Test if there are CUDA devices."""
        self.assertGreater(torch.cuda.device_count(), 0, "No CUDA devices found")
        
    def test_tensor_on_cuda(self):
        """Test if a tensor can be moved to CUDA and operations work."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping tensor test")
            
        # Create a tensor and move it to CUDA
        x = torch.rand(5, 3)
        x_cuda = x.cuda()
        
        # Verify it's on CUDA
        self.assertTrue(x_cuda.is_cuda, "Tensor not moved to CUDA")
        
        # Perform a simple operation
        y_cuda = x_cuda * 2
        
        # Check result
        self.assertTrue(y_cuda.is_cuda, "Result tensor not on CUDA")
        y_cpu = y_cuda.cpu()
        expected = x * 2
        
        self.assertTrue(torch.allclose(y_cpu, expected), "CUDA computation incorrect")

if __name__ == "__main__":
    unittest.main() 