# SPDX-License-Identifier: BSD-3-Clause
"""
Unit tests for sparse instance support.

Tests cover:
- Sparsity detection
- Dense/Sparse instance creation
- Automatic storage selection
- Configuration management
- Edge cases
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

# Import modules to test
from openmoa._sparse_utils import (
    SparseConfig,
    calculate_sparsity,
    should_use_sparse,
    create_dense_java_instance,
    create_sparse_java_instance,
    create_java_instance,
    get_storage_info,
)

# Try to import scipy
try:
    import scipy.sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestSparsityDetection(unittest.TestCase):
    """Test sparsity calculation and detection."""
    
    def setUp(self):
        """Reset configuration before each test."""
        SparseConfig.enabled = True
        SparseConfig.sparsity_threshold = 0.9
        SparseConfig.min_dimension = 100
        SparseConfig.max_nonzero = 10000
    
    def test_calculate_sparsity_dense_array(self):
        """Test sparsity calculation on dense arrays."""
        # 80% sparse
        x = np.array([0, 0, 0, 0, 1.0])
        self.assertAlmostEqual(calculate_sparsity(x), 0.8)
        
        # 100% sparse (all zeros)
        x = np.zeros(100)
        self.assertAlmostEqual(calculate_sparsity(x), 1.0)
        
        # 0% sparse (no zeros)
        x = np.ones(100)
        self.assertAlmostEqual(calculate_sparsity(x), 0.0)
    
    @unittest.skipIf(not SCIPY_AVAILABLE, "scipy not available")
    def test_calculate_sparsity_scipy_sparse(self):
        """Test sparsity calculation on scipy sparse matrices."""
        from scipy.sparse import csr_matrix
        
        # Create sparse matrix
        x = csr_matrix([[0, 0, 0.5, 0, 0]])
        sparsity = calculate_sparsity(x)
        self.assertAlmostEqual(sparsity, 0.8)  # 4 zeros out of 5
    
    def test_should_use_sparse_low_dimension(self):
        """Low dimension should prefer dense."""
        x = np.zeros(50)  # Below min_dimension
        x[0] = 1.0
        
        self.assertFalse(should_use_sparse(x))
    
    def test_should_use_sparse_high_dimension_high_sparsity(self):
        """High dimension + high sparsity should use sparse."""
        x = np.zeros(10000)  # High dimension
        x[[0, 10, 100]] = [1.0, 2.0, 3.0]  # 99.97% sparse
        
        self.assertTrue(should_use_sparse(x))
    
    def test_should_use_sparse_low_sparsity(self):
        """Low sparsity should use dense."""
        x = np.random.rand(1000)  # ~0% sparse
        
        self.assertFalse(should_use_sparse(x))
    
    def test_should_use_sparse_too_many_nonzero(self):
        """Too many non-zero elements should use dense."""
        x = np.zeros(100000)
        x[:20000] = 1.0  # 20,000 non-zero (exceeds max_nonzero)
        
        self.assertFalse(should_use_sparse(x))
    
    def test_force_sparse_override(self):
        """force=True should override automatic detection."""
        x = np.random.rand(50)  # Would normally be dense
        
        self.assertTrue(should_use_sparse(x, force=True))
    
    def test_force_dense_override(self):
        """force=False should override automatic detection."""
        x = np.zeros(10000)
        x[0] = 1.0  # Would normally be sparse
        
        self.assertFalse(should_use_sparse(x, force=False))
    
    def test_config_disable_auto_sparse(self):
        """Disabling auto-sparse should always return False."""
        SparseConfig.disable_auto_sparse()
        
        x = np.zeros(10000)
        x[0] = 1.0  # High sparsity, would normally use sparse
        
        self.assertFalse(should_use_sparse(x))
        
        # Reset
        SparseConfig.enable_auto_sparse()
    
    def test_config_set_threshold(self):
        """Changing threshold should affect detection."""
        SparseConfig.set_threshold(0.95)
        
        x = np.zeros(1000)
        x[:60] = 1.0  # 94% sparse (below new threshold)
        
        self.assertFalse(should_use_sparse(x))
        
        # Reset
        SparseConfig.set_threshold(0.9)
    
    def test_config_invalid_threshold(self):
        """Invalid threshold should raise error."""
        with self.assertRaises(ValueError):
            SparseConfig.set_threshold(1.5)
        
        with self.assertRaises(ValueError):
            SparseConfig.set_threshold(-0.1)


class TestJavaInstanceCreation(unittest.TestCase):
    """Test creation of MOA Java instances."""
    
    def setUp(self):
        """Create shared schemas for tests."""
        from openmoa.stream import Schema
        
        # Small schema for dense tests (3 features)
        self.schema_small = Schema.from_custom(
            ["f1", "f2", "f3"],
            dataset_name="TestDataset",
            values_for_class_label=["A", "B"]
        )
        
        # Medium schema for sparse tests (5 features)
        self.schema_medium = Schema.from_custom(
            ["f1", "f2", "f3", "f4", "f5"],
            dataset_name="TestDataset",
            values_for_class_label=["A", "B"]
        )
        
        # Large schema for auto-detection tests (10000 features)
        self.schema_large = Schema.from_custom(
            [f"f{i}" for i in range(10000)],
            dataset_name="LargeDataset",
            values_for_class_label=["A", "B"]
        )
        
        # Small schema for random tests (50 features)
        self.schema_50 = Schema.from_custom(
            [f"f{i}" for i in range(50)],
            dataset_name="SmallDataset",
            values_for_class_label=["A", "B"]
        )
    
    def test_create_dense_instance(self):
        """Test creating DenseInstance."""
        x = np.array([0.1, 0.2, 0.3])
        instance = create_dense_java_instance(x)
        
        # Verify type
        from com.yahoo.labs.samoa.instances import DenseInstance
        self.assertIsInstance(instance, DenseInstance)
        
        # ✅ Set dataset before accessing attributes
        instance.setDataset(self.schema_small.get_moa_header())
        
        # Verify size (3 features + 1 class)
        self.assertEqual(instance.numAttributes(), 4)
        self.assertEqual(instance.numInputAttributes(), 3)
        
        # Verify values
        self.assertAlmostEqual(instance.value(0), 0.1)
        self.assertAlmostEqual(instance.value(1), 0.2)
        self.assertAlmostEqual(instance.value(2), 0.3)
    
    def test_create_sparse_instance(self):
        """Test creating SparseInstance."""
        x = np.array([0, 0, 0.5, 0, 0.3])
        instance = create_sparse_java_instance(x)
        
        # Verify type
        from com.yahoo.labs.samoa.instances import SparseInstance
        self.assertIsInstance(instance, SparseInstance)
        
        # ✅ Set dataset before accessing attributes
        instance.setDataset(self.schema_medium.get_moa_header())
        print("\n=== Debug SparseInstance ===")
        print(f"numAttributes: {instance.numAttributes()}")
        print(f"numInputAttributes: {instance.numInputAttributes()}")
        print(f"numValues: {instance.numValues()}")
        for i in range(instance.numInputAttributes()):
            print(f"  value({i}) = {instance.value(i)}")
        print("============================\n")       
        # Verify size
        self.assertEqual(instance.numAttributes(), 6)  # 5 features + 1 class
        self.assertEqual(instance.numInputAttributes(), 5)
        
        # Verify only non-zero values are stored (+ class slot)
        self.assertEqual(instance.numValues(), 3)  # 2 non-zero + 1 class slot
        
        # Verify values
        self.assertAlmostEqual(instance.value(2), 0.5)
        self.assertAlmostEqual(instance.value(4), 0.3)
        self.assertAlmostEqual(instance.value(0), 0.0)  # Zero value
    
    def test_create_dense_instance_empty(self):
        """Empty array should raise error."""
        x = np.array([])
        
        with self.assertRaises(ValueError):
            create_dense_java_instance(x)
    
    def test_create_dense_instance_2d(self):
        """2D array should raise error."""
        x = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        with self.assertRaises(ValueError):
            create_dense_java_instance(x)
    
    def test_create_sparse_instance_all_zeros(self):
        """All-zero array should raise error."""
        x = np.zeros(100)
        
        with self.assertRaises(ValueError):
            create_sparse_java_instance(x)
    
    def test_create_java_instance_automatic_dense(self):
        """Automatic detection should choose dense."""
        x = np.random.rand(50)  # Low dimension, not sparse
        instance = create_java_instance(x)
        
        from com.yahoo.labs.samoa.instances import DenseInstance
        self.assertIsInstance(instance, DenseInstance)
    
    def test_create_java_instance_automatic_sparse(self):
        """Automatic detection should choose sparse."""
        x = np.zeros(10000)
        x[[42, 123]] = [0.5, 0.3]  # High dimension, high sparsity
        instance = create_java_instance(x)
        
        from com.yahoo.labs.samoa.instances import SparseInstance
        self.assertIsInstance(instance, SparseInstance)
    
    def test_create_java_instance_force_sparse(self):
        """Force sparse should create SparseInstance."""
        x = np.random.rand(50)  # Would be dense normally
        instance = create_java_instance(x, force_sparse=True)
        
        from com.yahoo.labs.samoa.instances import SparseInstance
        self.assertIsInstance(instance, SparseInstance)
    
    def test_create_java_instance_force_dense(self):
        """Force dense should create DenseInstance."""
        x = np.zeros(10000)
        x[0] = 1.0  # Would be sparse normally
        instance = create_java_instance(x, force_sparse=False)
        
        from com.yahoo.labs.samoa.instances import DenseInstance
        self.assertIsInstance(instance, DenseInstance)


class TestStorageInfo(unittest.TestCase):
    """Test storage information utility."""
    
    def test_get_storage_info(self):
        """Test storage info calculation."""
        x = np.zeros(10000)
        x[[0, 100, 1000]] = [1.0, 2.0, 3.0]  # 3 non-zero
        
        info = get_storage_info(x)
        
        # Check keys exist
        self.assertIn('num_features', info)
        self.assertIn('num_nonzero', info)
        self.assertIn('sparsity', info)
        self.assertIn('dense_memory_bytes', info)
        self.assertIn('sparse_memory_bytes', info)
        self.assertIn('memory_saving_ratio', info)
        self.assertIn('recommended', info)
        
        # Check values
        self.assertEqual(info['num_features'], 10000)
        self.assertEqual(info['num_nonzero'], 3)
        self.assertAlmostEqual(info['sparsity'], 0.9997)
        
        # Dense: 10000 * 8 = 80,000 bytes
        self.assertEqual(info['dense_memory_bytes'], 80000)
        
        # Sparse: 3 * 12 = 36 bytes
        self.assertEqual(info['sparse_memory_bytes'], 36)
        
        # Saving ratio
        expected_saving = (80000 - 36) / 80000
        self.assertAlmostEqual(info['memory_saving_ratio'], expected_saving)
        
        # Should recommend sparse
        self.assertEqual(info['recommended'], 'sparse')


if __name__ == '__main__':
    unittest.main()