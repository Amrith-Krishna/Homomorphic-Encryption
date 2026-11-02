import numpy as np
from Pyfhel import Pyfhel  
import time
import pandas as pd

def digits_of_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Arrays must have the same shape: {a.shape} vs {b.shape}")
    nonzero_mask = b != 0
    a_valid = a[nonzero_mask]
    b_valid = b[nonzero_mask]
    relative_error = np.abs(a_valid - b_valid) / np.abs(b_valid)
    exact_mask = relative_error == 0
    sig_digits = np.where(exact_mask,16,-np.log10(relative_error + 1e-300))
    return float(np.mean(sig_digits))

rng = np.random.default_rng()
data={}
bfv=[]
bfv50=[]
ckks=[]
ckks2=[]
ckksbfv=[]
ub=[]
n=12
l=12

for i in range(100):
    lb=-25+0.5*i
    ub.append(lb)
    dlb=10**lb
    vec_x = np.array([rng.uniform(10**-30, dlb) for n in range(2**l)], dtype=np.float64)
    vec_y = np.array([rng.uniform(10**-30, dlb) for n in range(2**l)], dtype=np.float64)
    out=vec_x*vec_y
    out=out.astype(np.float64)
    # ckks

    HE_ckks = Pyfhel()
    HE_ckks.contextGen(scheme='CKKS', n=2**13, scale=2**30, qi_sizes=[60,30,30,60])
    HE_ckks.keyGen()             
    time_ckks_only = 0

    start = time.perf_counter_ns()
    cx = HE_ckks.encrypt(vec_x)
    cy = HE_ckks.encrypt(vec_y)
    result = HE_ckks.decryptFrac(cx*cy)
    time_ckks_only = time.perf_counter_ns() - start
    result = result[0:2**l]

    err_avg = max(digits_of_accuracy(result, out),0)
    ckks.append(err_avg)
    print("CKKS",err_avg)

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    # bfv_2

    HE_m = Pyfhel()
    HE_m.contextGen(scheme='BFV', n=2**n, t_bits=16, sec=128)
    HE_m.keyGen()
    HE_e = Pyfhel()
    HE_e.contextGen(scheme='BFV', n=2**n, t_bits=16, sec=128)
    HE_e.keyGen()

    time_op=0
    start = time.perf_counter_ns()
    mant, exp = np.frexp(vec_x)
    mant = (np.log2(np.abs(mant))*(2**13)).astype(np.int64)
    mant2, exp2 = np.frexp(vec_y)
    mant2 = (np.log2(np.abs(mant2))*2**13).astype(np.int64)
    time_op = time.perf_counter_ns()-start

    time_bfv1=0
    start = time.perf_counter_ns()
    cm_x = HE_m.encrypt(mant)
    cm_y = HE_m.encrypt(mant2)
    cm_res = cm_x+cm_y
    m_res = HE_m.decrypt(cm_res)/2**13
    time_bfv1 = time.perf_counter_ns()-start

    time_bfv2=0
    start = time.perf_counter_ns()
    ce_x = HE_e.encrypt(exp)
    ce_y = HE_e.encrypt(exp2)
    ce_res = ce_x+ce_y
    e_res = HE_e.decrypt(ce_res)
    time_bfv2 = time.perf_counter_ns()-start

    start = time.perf_counter_ns()
    result = np.ldexp(np.exp2(m_res[:min(2**l, 2**12)]), e_res[:min(2**n, 2**l)])
    result = result[0:2**l]
    time_op += time.perf_counter_ns() - start

    err_avg = max(digits_of_accuracy(result, out),0)
    bfv.append(err_avg)
    print("BFV2",err_avg)
    
    
    HE_m = Pyfhel()
    HE_m.contextGen(scheme='BFV', n=2**n, t_bits=55, sec=128)
    HE_m.keyGen()
    HE_e = Pyfhel()
    HE_e.contextGen(scheme='BFV', n=2**n, t=65537, sec=128)
    HE_e.keyGen()

    time_op=0
    start = time.perf_counter_ns()
    mant, exp = np.frexp(vec_x)
    mant = (np.log2(np.abs(mant))*(2**53)).astype(np.int64)
    mant2, exp2 = np.frexp(vec_y)
    mant2 = (np.log2(np.abs(mant2))*2**53).astype(np.int64)
    time_op = time.perf_counter_ns()-start

    time_bfv1=0
    start = time.perf_counter_ns()
    cm_x = HE_m.encrypt(mant)
    cm_y = HE_m.encrypt(mant2)
    cm_res = cm_x+cm_y
    m_res = HE_m.decrypt(cm_res)/2**53
    time_bfv1 = time.perf_counter_ns()-start

    time_bfv2=0
    start = time.perf_counter_ns()
    ce_x = HE_e.encrypt(exp)
    ce_y = HE_e.encrypt(exp2)
    ce_res = ce_x+ce_y
    e_res = HE_e.decrypt(ce_res)
    time_bfv2 = time.perf_counter_ns()-start

    start = time.perf_counter_ns()
    result = np.ldexp(np.exp2(m_res[:min(2**l, 2**12)]), e_res[:min(2**n, 2**l)])
    result = result[0:2**l]
    time_op += time.perf_counter_ns() - start

    err_avg = max(digits_of_accuracy(result, out),0)
    bfv50.append(err_avg)
    print("BFV50",err_avg)
    
    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    # ckks_2

    time_op=0
    time_ckks1=0

    HE_m = Pyfhel()
    HE_m.contextGen(scheme='CKKS', n=2**13, scale=2**30, qi_sizes=[60,30,30,60])
    HE_m.keyGen()

    start = time.perf_counter_ns()
    mant = np.log2(vec_x)
    mant2 = np.log2(vec_y)
    time_op = time.perf_counter_ns()-start

    start = time.perf_counter_ns()
    cm_x = HE_m.encrypt(mant)
    cm_y = HE_m.encrypt(mant2)
    cm_res = cm_x+cm_y
    m_res = HE_m.decryptFrac(cm_res)
    time_ckks1=time.perf_counter_ns()-start

    start = time.perf_counter_ns()
    m_res = m_res[:min(2**n,2**l)]
    m_res = np.exp2(m_res)
    result = m_res[0:2**l]
    time_op += time.perf_counter_ns() - start
    err_avg = max(digits_of_accuracy(result, out),0)
    ckks2.append(err_avg)
    print("CKKS2",err_avg)
    print(result[0],out[0])
    
    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    # bfv_ckks
    HE_ckks = Pyfhel()
    HE_ckks.contextGen(scheme='CKKS', n=2**13, scale=2**30, qi_sizes=[60,30,30,60])
    HE_ckks.keyGen();

    HE_bfv = Pyfhel()
    HE_bfv.contextGen(scheme='BFV', n=2**n, t=65537, sec=128); HE_bfv.keyGen()

    time_ckks = 0
    time_bfv = 0
    time_op = 0

    start = time.perf_counter_ns()
    mant, exp = np.frexp(vec_x)  # Decompose: mant in [0.5,1), exp int
    mant2, exp2 = np.frexp(vec_y)  # Decompose: mant in [0.5,1), exp int
    exp = exp.astype(np.int64)
    exp2 = exp2.astype(np.int64)
    time_op+= time.perf_counter_ns()-start

    start = time.perf_counter_ns()
    ctx_x = HE_ckks.encrypt(mant)
    ctx_y = HE_ckks.encrypt(mant2)
    ct_res_m = ctx_x*ctx_y
    mant_raw = HE_ckks.decrypt(ct_res_m) 
    time_ckks += time.perf_counter_ns()-start

    start = time.perf_counter_ns()
    cte_x = HE_bfv.encrypt(exp)
    cte_y = HE_bfv.encrypt(exp2)
    ct_res_e = cte_x+cte_y
    exp = HE_bfv.decrypt(ct_res_e)
    time_bfv += time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    result = np.ldexp(mant_raw[:min(2**l, 2**12)], exp[:min(2**n, 2**l)].astype(int))
    result = result[0:2**l]
    time_op += time.perf_counter_ns() - start
    err_avg = max(digits_of_accuracy(result, out),0)
    ckksbfv.append(err_avg)
    print("BFVCKKS",err_avg)

    
data["UB"]=ub
data["BFV+BFV t_bits=16"]=bfv
data["BFV+BFV t_bits=55"]=bfv50
data["CKKS"]=ckks
data["CKKS+BFV"]=ckksbfv
data["CKKS+CKKS"]=ckks2

df=pd.DataFrame(data)
df.to_csv('final_prec_v2.csv',index=False)