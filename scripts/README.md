# Scripts Directory

This directory contains all scripts and utilities for the Competitive Intelligence v2 system, organized by purpose.

## 📁 Directory Structure

```
scripts/
├── README.md                     # This file
├── qa-validation/               # QA validation and testing scripts
│   ├── qa_validation.py         # Main comprehensive QA validation
│   ├── focused_qa_test.py       # Focused testing for specific fixes
│   ├── simple_fix_validation.py # Simple endpoint validation
│   ├── qa_validation_results.json # Latest QA results
│   └── focused_qa_results.json  # Focused test results
└── analysis/                    # Code analysis and metrics scripts
    └── analyze_codebase.py      # Codebase analysis and metrics
```

---

## 🧪 QA Validation Scripts

### 🔍 `qa-validation/qa_validation.py` - **MAIN QA SCRIPT**
**Primary validation script for complete system testing with 100% success rate.**

**What it tests:**
- ✅ Authentication Module (4 endpoints)
- ✅ User Management Module (2 endpoints)  
- ✅ Strategic Profile Module (3 endpoints)
- ✅ Focus Areas Module (3 endpoints)
- ✅ Entity Tracking Module (4 endpoints)
- ✅ Delivery Preferences Module (3 endpoints)
- ✅ **Total: 19 endpoints across 6 modules**

**Current Status**: **100% SUCCESS RATE (19/19 tests passing)**

**Usage:**
```bash
cd "C:\Users\ceato\competitive-intel-v2"
set RATE_LIMIT_ENABLED=False && python scripts/qa-validation/qa_validation.py
```

**Output Example:**
```
================================================================================
COMPREHENSIVE QA VALIDATION - USER CONFIG SERVICE
================================================================================
Target: http://127.0.0.1:8002
Success Rate: 100.0% (19/19)
Overall Status: PASS
================================================================================
```

### 🎯 `qa-validation/focused_qa_test.py` - **Focused Fix Testing**
**Specialized testing for the three major fixes implemented.**

**What it tests:**
- ✅ Fix #1: Delivery preferences schema validation
- ✅ Fix #2: Missing /me authentication endpoint  
- ✅ Fix #3: Entity tracking authentication issues

**Usage:**
```bash
python scripts/qa-validation/focused_qa_test.py
```

### ⚡ `qa-validation/simple_fix_validation.py` - **Quick Health Check**
**Simplified validation for rapid system health verification.**

**What it tests:**
- Core endpoint availability (200/401 responses)
- Authentication requirement validation
- API documentation accessibility

**Usage:**
```bash
python scripts/qa-validation/simple_fix_validation.py
```

---

## 📊 Analysis Scripts

### 🔬 `analysis/analyze_codebase.py` - **Codebase Analysis**
**Comprehensive codebase metrics and analysis tool.**

**What it analyzes:**
- Lines of code across modules
- Import dependency analysis
- Code complexity metrics
- File organization structure
- Optimization opportunities

**Usage:**
```bash
python scripts/analysis/analyze_codebase.py
```

---

## 📋 Results Files

### 🧪 QA Results
- **`qa-validation/qa_validation_results.json`** - Latest comprehensive QA results with full metrics
- **`qa-validation/focused_qa_results.json`** - Focused fix testing results

**Example QA Results:**
```json
{
  "timestamp": "2025-08-20T22:06:57",
  "total_duration_ms": 10693.58,
  "summary": {
    "total_modules": 6,
    "modules_passed": 6,
    "total_tests": 19,
    "tests_passed": 19,
    "tests_failed": 0
  }
}
```

---

## 🚀 Usage Guidelines

### **For Developers**

#### Before Committing Code
```bash
python scripts/qa-validation/simple_fix_validation.py
```

#### Before Major Changes
```bash
set RATE_LIMIT_ENABLED=False && python scripts/qa-validation/qa_validation.py
```

#### For Code Analysis
```bash
python scripts/analysis/analyze_codebase.py
```

### **For QA Teams**

#### Pre-Release Validation
1. Run comprehensive QA: `python scripts/qa-validation/qa_validation.py`
2. Verify 100% success rate
3. Check performance metrics (<1000ms average)
4. Document any failures with timestamps

#### Post-Fix Validation
1. Run focused tests: `python scripts/qa-validation/focused_qa_test.py`
2. Verify specific fixes are working
3. Run full validation to ensure no regressions

### **For DevOps/Deployment**

#### Production Readiness Check
```bash
# Full system validation
set RATE_LIMIT_ENABLED=False && python scripts/qa-validation/qa_validation.py

# Expected output: 100% success rate
# If not 100%, DO NOT DEPLOY
```

#### Health Monitoring
```bash
# Quick health check
python scripts/qa-validation/simple_fix_validation.py

# Should return all endpoints accessible with proper auth
```

---

## 📈 Performance Benchmarks

### **Current System Performance** (Post-Optimization)
- **Average Response Time**: ~563ms
- **Authentication Module**: ~686ms avg
- **User Management**: ~537ms avg  
- **Strategic Profile**: ~474ms avg
- **Focus Areas**: ~525ms avg
- **Entity Tracking**: ~590ms avg
- **Delivery Preferences**: ~504ms avg

### **QA Validation Performance**
- **Total Validation Time**: ~10.7 seconds for 19 tests
- **Success Rate**: 100% (19/19 tests passing)
- **No Critical Failures**: All endpoints functional

---

## 🔧 Troubleshooting

### **Common Issues**

#### Rate Limiting Errors (429)
```bash
# Solution: Disable rate limiting for testing
set RATE_LIMIT_ENABLED=False && python script_name.py
```

#### Server Not Running
```bash
# Start the server first
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload
```

#### Import Errors
```bash
# Ensure virtual environment is activated
venv\Scripts\activate
pip install -r requirements.txt
```

### **Expected Behaviors**

#### Authentication Tests
- **401 Unauthorized**: Expected for endpoints without valid tokens
- **200 OK**: Expected for valid authenticated requests
- **404 Not Found**: Should NOT occur for implemented endpoints

#### Performance Tests
- **Sub-1000ms**: All endpoints should respond within 1 second
- **Consistent Results**: Multiple runs should show similar performance

---

## 🎯 Integration with Development Workflow

### **Git Hooks** (Recommended)
```bash
# Pre-commit hook
#!/bin/bash
cd "C:\Users\ceato\competitive-intel-v2"
python scripts/qa-validation/simple_fix_validation.py
if [ $? -ne 0 ]; then
    echo "QA validation failed. Commit blocked."
    exit 1
fi
```

### **CI/CD Pipeline**
```yaml
- name: Run Comprehensive QA
  run: |
    cd competitive-intel-v2
    set RATE_LIMIT_ENABLED=False
    python scripts/qa-validation/qa_validation.py
    
- name: Verify 100% Success Rate
  run: |
    # Check QA results for 100% success
    grep "SUCCESS RATE: 100.0%" qa_validation_results.json
```

---

## 📚 Script Documentation

### **QA Validation Scripts**
- **Comprehensive**: Full system validation across all 6 modules
- **Focused**: Specific fix validation for critical issues
- **Simple**: Quick health checks for rapid feedback

### **Analysis Scripts**  
- **Codebase Analysis**: Metrics, dependencies, and optimization opportunities

### **Results Tracking**
- **JSON Results**: Machine-readable results for automation
- **Human-Readable**: Clear ASCII output for manual review

---

**All scripts maintain ASCII-only output for maximum compatibility and can be run in any environment.**

*Scripts Directory Updated: 2025-08-20*  
*Current QA Success Rate: 100% (19/19 tests passing)*  
*Total Scripts: 6 organized across 2 categories*