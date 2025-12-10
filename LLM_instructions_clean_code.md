# LLM Instructions for Code Cleanup

This document provides instructions for an AI agent to clean up and optimize the Swedish Handwritten OCR project codebase.

## Project Context

**Purpose**: Swedish handwritten line-level text recognition using TrOCR  
**Current Status**: Production-ready system with complete pipeline  
**Architecture**: Three-processor system (LinePreprocessor, TextFieldPreprocessor, legacy ImagePreprocessor)

## High-Priority Cleanup Tasks

### 1. Remove Obsolete Synthetic Data Module
**Location**: `scripts/data_processing/synthetic_data/`  
**Action**: 
- DELETE `line_generator.py` (word-to-line conversion hack, no longer needed)
- KEEP `synthetic_data_creator.py` (font-based generator, still valuable)  
- UPDATE `__init__.py` to only export `generate_synthetic_data`
- VERIFY orchestrator no longer references `line_generator.py`

**Verification**: Ensure orchestrator pipeline runs without errors after removal

### 2. Update Legacy Comments and Docstrings
**Target**: Remove "LEGACY", "TODO", "WILL BE UPDATED" markers throughout codebase  
**Focus Areas**:
- `scripts/data_processing/image_segmentation/` - Remove word-level legacy comments
- `scripts/data_processing/template_generator/` - Update to reflect current dual-format support
- `scripts/data_processing/orchestrator/` - Remove synthetic data references

**Pattern to Find**: `"LEGACY"`, `"TODO"`, `"will be updated"`, `"future versions"`  
**Action**: Remove or update to reflect current capabilities

### 3. Consolidate Import Statements
**Target**: Clean up unused imports across all Python files  
**Common Issues Found**:
- Unused `Tuple` imports in several files
- Redundant `Path` imports when `pathlib.Path` already imported
- Unused `logging` imports in files that don't log

**Tool**: Use automated import cleanup tools or manual review

### 4. Standardize Error Handling
**Target**: Ensure consistent error handling patterns  
**Current Pattern**: Try/except with logger.error and re-raise  
**Focus**: 
- `scripts/data_processing/orchestrator/` modules
- `scripts/training/` modules
- `scripts/evaluation/` modules

### 5. Remove Dead Code Paths
**Areas to Check**:
- Word-level processing code that's no longer accessible
- Conditional branches for old dataset versions (v1)
- Debug code that prints unnecessary information
- Functions that are defined but never called

### 6. Update Configuration Comments
**File**: `config/config.yaml`  
**Action**: Ensure comments reflect line-level focus, not word-level

## Medium-Priority Cleanup Tasks

### 7. Optimize File I/O Operations
**Focus**: Reduce redundant file reads/writes in orchestrator pipeline  
**Check**: Multiple reads of same metadata files, unnecessary file existence checks

### 8. Consolidate Path Management
**File**: `config/paths.py`  
**Action**: Remove any unused path definitions, ensure all paths are actually used

### 9. Standardize Logging Levels
**Pattern**: Ensure DEBUG, INFO, WARNING, ERROR are used consistently  
**Common Issue**: Too many INFO logs that should be DEBUG

### 10. Clean Up Test/Debug Code
**Pattern to Find**: `print()` statements that should be `logger.debug()`  
**Action**: Replace with proper logging or remove entirely

## Low-Priority Cleanup Tasks

### 11. Type Hints Completion
**Target**: Add type hints where missing, especially in newer modules  
**Focus**: Public functions and class methods

### 12. Docstring Standardization
**Format**: Use consistent Google-style or NumPy-style docstrings  
**Focus**: Public APIs and complex functions

### 13. Code Formatting
**Tool**: Use black or similar formatter for consistent code style  
**Config**: Follow PEP 8 standards

## Files to Preserve (Do Not Modify)

### Core Production Files
- `scripts/training/train_model.py` - Validated training pipeline
- `scripts/evaluation/evaluate_model.py` - Working evaluation system
- `scripts/data_processing/orchestrator/main.py` - Core orchestrator logic
- `config/paths.py` - Critical path management
- `requirements.txt` - Dependencies

### Template and Data Files
- `docs/data_collection/line_texts/*.json` - Swedish text collections
- `models/final_model/` - Trained models
- `dataset/` directory structure - Training data

## Verification Steps

After cleanup, verify:

1. **Pipeline functionality**:
   ```bash
   python -m scripts.data_processing.orchestrator.main --dry-run
   python -m scripts.training.train_model --dry_run --epochs 1
   python -m scripts.evaluation.evaluate_model
   ```

2. **Import integrity**:
   ```bash
   python -c "import scripts.data_processing.orchestrator.main"
   python -c "import scripts.training.train_model"
   python -c "import scripts.evaluation.evaluate_model"
   ```

3. **No broken references**: Search for any remaining references to deleted modules

## Code Quality Standards

### Naming Conventions
- Classes: PascalCase
- Functions/methods: snake_case  
- Constants: UPPER_SNAKE_CASE
- Files/modules: snake_case.py

### Error Messages
- Include context (file paths, writer IDs, etc.)
- Use logger.error() for user-facing errors
- Use logger.debug() for developer information

### Performance Considerations
- Avoid redundant file I/O operations
- Use generators for large dataset iteration
- Minimize memory footprint in image processing

## Expected Outcomes

After cleanup:
- **Reduced codebase size**: ~10-15% reduction in lines of code
- **Improved maintainability**: Clear separation between current and legacy functionality
- **Better performance**: Reduced I/O operations and memory usage
- **Enhanced readability**: Consistent formatting and clear documentation
- **Production readiness**: Remove all experimental/temporary code

## Risk Mitigation

### Backup Strategy
1. Create full backup before starting cleanup
2. Test functionality after each major change
3. Commit changes incrementally, not as single large commit

### Rollback Plan
- Keep backup of removed files in separate directory
- Document all changes for potential reversal
- Maintain git history for easy rollback if needed

## Success Criteria

- [ ] All TODO/LEGACY markers removed or updated
- [ ] No unused imports remaining
- [ ] Consistent error handling patterns
- [ ] All dead code paths removed
- [ ] Pipeline tests pass successfully
- [ ] Performance improved (measurable reduction in execution time)
- [ ] Code passes linting tools (pylint, flake8)
- [ ] Documentation reflects actual codebase state

This cleanup will result in a production-ready, maintainable codebase optimized for Swedish handwritten line-level OCR processing.