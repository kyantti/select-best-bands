#!/usr/bin/env python3
"""
Test script to verify the separated modules work correctly
"""


def test_data_loading():
    """Test the data loading module"""
    print("🧪 Testing data loading module...")
    try:
        from cnn.going_modular.data_setup import load_hypercubes_to_memory

        hypercubes = load_hypercubes_to_memory()

        print(
            f"✅ Successfully loaded {sum(len(patches) for patches in hypercubes.values())} patches"
        )
        print(f"📊 Classes: {list(hypercubes.keys())}")
        return hypercubes
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None


def test_cnn_evaluation():
    """Test the CNN evaluation function"""
    print("\n🧪 Testing CNN evaluation module...")
    try:
        from cnn.transfer_learning import evaluate_band_combination_fitness

        # Load data
        hypercubes = test_data_loading()
        if hypercubes is None:
            return False

        # Test with a simple band combination
        print("🎨 Testing band combination [60, 30, 10]...")
        fitness = evaluate_band_combination_fitness(
            hypercubes,
            60,
            30,
            10,
            epochs=2,  # Very short test
            batch_size=16,
            verbose=True,
        )

        print(f"✅ CNN evaluation successful! Fitness: {fitness:.4f}")
        return True
    except Exception as e:
        print(f"❌ CNN evaluation failed: {e}")
        return False


def test_genetic_algorithm():
    """Test the genetic algorithm module"""
    print("\n🧪 Testing genetic algorithm module...")
    try:
        from ga import main

        print("🧬 Running mini GA test...")
        best_individual, best_fitness = main()

        print("✅ GA test successful!")
        print(f"🥇 Best bands: {list(best_individual)}")
        print(f"🎯 Best fitness: {best_fitness:.4f}")
        return True
    except Exception as e:
        print(f"❌ GA test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing separated modules...")
    print("=" * 50)

    # Test each module
    success_count = 0

    if test_data_loading():
        success_count += 1

    if test_cnn_evaluation():
        success_count += 1

    if test_genetic_algorithm():
        success_count += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/3 modules working")

    if success_count == 3:
        print("🎉 All modules working correctly!")
        print(
            "✅ You can now run the genetic algorithm with: python genetic_algorithm.py"
        )
    else:
        print("❌ Some modules have issues. Check the error messages above.")
