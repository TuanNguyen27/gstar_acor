# ACOR: Which Files to Use

**Quick Guide for Reviewers and Users**

---

## ✅ PRODUCTION FILES (Use These!)

### Evaluation Scripts
```
evaluation/
├── comprehensive_8bit_eval.py        ← MAIN EVALUATION (FIXED!)
├── diagnose_single_problem.py        ← DEBUG TOOL (FIXED!)
└── test_attention_only_adapters.py   ← VERIFY ADAPTERS WORK
```

**Key Updates (Oct 4, 2025)**:
- ✅ Planner token limit: 100 → 200 (no more truncation!)
- ✅ Answer extraction: Standard GSM8K method
- ✅ Adapter swapping: 50% memory reduction

### Training Scripts
```
training/
├── train_planner_attention_only.py   ← PLANNER (9.7K examples, attention-only LoRA)
└── train_executor_attention_only.py  ← EXECUTOR (11K examples, attention-only LoRA)
```

**Trained Models (Modal Volume)**:
```
/vol/training_runs/
├── planner_attention_only_20251003_170348/final_model/   ← USE THIS
└── executor_attention_only_20251003_170711/final_model/  ← USE THIS
```

### Documentation
```
├── README.md                         ← UPDATED (Oct 4) - Project overview
├── EVALUATION_SUMMARY.md             ← NEW (Oct 4) - Complete analysis
└── FILES_TO_USE.md                   ← THIS FILE - Quick reference
```

---

## ⚠️ DEPRECATED FILES (Don't Use)

### Old Evaluation Configs
```
evaluation/
├── comprehensive_baseline_eval.py    ← OLD (wrong settings)
├── comprehensive_balanced_eval.py    ← OLD (wrong settings)
├── comprehensive_fp16_merged_eval.py ← FAILED (OOM issues)
└── *_standalone_eval.py             ← OLD (single-module tests)
```

### Failed Experiments
```
evaluation/
├── test_dora_workarounds.py         ← DoRA didn't work with 4-bit
├── test_lora_adapters.py            ← Debugging script (not for production)
├── test_executor_adapter.py         ← One-off test
├── test_planner_adapter.py          ← One-off test
└── inspect_adapter_weights.py       ← Debugging only
```

### Old Training Scripts
```
training/
├── train_planner_lora.py            ← OLD (MLP layers included - broken!)
├── train_executor_full.py           ← OLD (MLP layers included - broken!)
├── fixed_planner_retraining.py      ← DEPRECATED
├── optimized_planner_retraining.py  ← DEPRECATED
└── *_dora*.py                       ← ALL DoRA attempts FAILED
```

---

## 📊 Quick Start Commands

### 1. Run Fixed Evaluation
```bash
cd evaluation
modal run --detach comprehensive_8bit_eval.py::main --num-problems=100

# Monitor progress
modal app logs <app-id>
```

**Expected Output**:
- Accuracy: 50-60% (vs 30% with truncated plans)
- Time: 25-35 minutes
- Cost: ~$5-10

### 2. Debug Specific Problems
```bash
# Problem 25 (Kyle's book - previously truncated)
modal run diagnose_single_problem.py --problem-id=25

# Problem 52 (Tom's ship - previously truncated)
modal run diagnose_single_problem.py --problem-id=52

# Janet's ducks (baseline test - always works)
modal run diagnose_single_problem.py --problem-id=0
```

### 3. Verify Adapters Still Work
```bash
modal run test_attention_only_adapters.py
```

---

## 📝 What Changed (Oct 3-4, 2025)

### October 3: Answer Extraction Fix
**Problem**: 2% accuracy due to extraction failures  
**Fix**: Implemented standard GSM8K method  
**Result**: 15x improvement (2% → 30%)

**Files Updated**:
- `comprehensive_8bit_eval.py`: `_extract_answer()` method (lines 558-598)
- `diagnose_single_problem.py`: `extract_answer_standard_gsm8k()` (lines 177-216)

### October 4: Planner Token Limit Fix
**Problem**: 70% failures due to plan truncation at 100 tokens  
**Fix**: Increased to 200 tokens  
**Expected Result**: 2x improvement (30% → 50-60%)

**Files Updated**:
- `comprehensive_8bit_eval.py`: 
  - Line 1-22: Updated header/docstring
  - Line 279: `max_tokens=200` (was 100)
  - Line 642: Updated print statement
  - Line 732: Updated config metadata
  
- `diagnose_single_problem.py`:
  - Line 113: `max_new_tokens=200` (was 100)
  - Line 101: Updated print statement

---

## 🔍 How to Verify the Fix Worked

### Diagnostic Checklist

**1. Plans no longer truncated**
```bash
modal run diagnose_single_problem.py --problem-id=25
# Look for: "PLAN GENERATED:" with complete steps (no "[TRUNCATED]")
```

**Expected Before Fix**:
```
PLAN GENERATED:
**Step 2: Convert... = 10 miles/h [TRUNCATED]
```

**Expected After Fix**:
```
PLAN GENERATED:
Step 1: Define problem
Step 2: Convert percentage to decimal
Step 3: Calculate original price = $19.50 / 0.75 = $26
Step 4: Verify answer
```

**2. Higher completion rate**
```bash
# Run full evaluation, monitor output
modal app logs <app-id>

# Look for pattern change:
# Before: ~30% problems with "Steps: 100%"
# After:  ~60-70% problems with "Steps: 100%"
```

**3. Final accuracy improvement**
```bash
# Check final results
# Before: 30/89 = 30% accuracy
# After:  Expected 45-55/89 = 50-60% accuracy
```

---

## 📞 Support & Troubleshooting

### Common Issues

**1. "Modal app not found"**
```bash
modal app list --env=main
# No apps running = evaluation finished or not started
```

**2. "Adapter not found"**
```bash
# Check adapters exist on Modal volume:
modal volume ls acor-training-vol training_runs/

# Should see:
# planner_attention_only_20251003_170348/
# executor_attention_only_20251003_170711/
```

**3. "Out of memory"**
```bash
# This should NOT happen anymore (adapter swapping fixes this)
# If it does, check you're using the FIXED evaluation script
# NOT the old baseline/balanced scripts
```

### Getting Help

1. **Read docs first**:
   - README.md (project overview)
   - EVALUATION_SUMMARY.md (detailed analysis)

2. **Check diagnostics**:
   ```bash
   modal run diagnose_single_problem.py --problem-id=0  # Janet's ducks (should always work)
   ```

3. **Verify setup**:
   ```bash
   modal app list --env=main  # Check running jobs
   modal volume ls acor-training-vol training_runs/  # Check adapters exist
   ```

---

## 🎯 Success Criteria

**You'll know the fix worked when:**

1. ✅ Plans are complete (no "[TRUNCATED]" in diagnostic output)
2. ✅ Step completion rate increases (30% → 60-70% of problems)
3. ✅ Final accuracy improves (30% → 50-60%)
4. ✅ Problems 25 & 52 now pass (previously failed despite "100% steps")

**Timeline:**
- Evaluation runtime: ~30 minutes
- Results analysis: ~5 minutes
- **Total time to validate fix: ~35 minutes**

---

**Summary**: Use `comprehensive_8bit_eval.py` and `diagnose_single_problem.py`. Everything else is either deprecated or for debugging only.

**Status**: ✅ Ready for validation  
**Last Updated**: October 4, 2025, 12:05 AM PST
