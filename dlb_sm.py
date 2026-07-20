import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool,shared_memory
import os

CPU_COUNT =os.cpu_count()
print("Detected CPU cores:",CPU_COUNT)

max_iter=100
tol = 1e-4

Test_p =[2,4,8,16,32,64,88]

SHM_NAME=None
SHM_SHAPE=None
SHM_DTYPE=None

def load_dataset(filename):
    data=np.loadtxt(filename)
    return data.astype(np.float32)

def get_output_filename(dataset):
    num=''.join(filter(str.isdigit,dataset))
    if num=="":
        num="0"
    return f"Out{num}.txt"

def SE(data,k):
    centroids=data[:k].copy()
    s=time.perf_counter()
    
    for _ in range(max_iter):
        dist=np.linalg.norm(data[:,None]-centroids,axis=2)
        labels=np.argmin(dist,axis=1)
        new_cent=np.zeros_like(centroids)

        for i in range(k):
            pts=data[labels==i]
            if len(pts)>0:
                new_cent[i]=pts.mean(axis=0)
            else:
                new_cent[i]=centroids[i]

        shift=np.linalg.norm(new_cent-centroids)
        centroids=new_cent

        if shift<tol:
            break

    total=time.perf_counter()-s
    return total

def assign_chunk(args):
    chunk,c=args
    dist=np.linalg.norm(chunk[:,None]-c,axis=2)
    return np.argmin(dist,axis=1)

def ECP(data,k):
    process=CPU_COUNT
    centroid=data[:k].copy()
    chunks=np.array_split(data,process)

    pool=Pool(process)
    start=time.perf_counter()

    for _ in range(max_iter):
        result=pool.map(assign_chunk,[(c,centroid) for c in chunks])
        labels=np.concatenate(result)
        new_centroid=np.zeros_like(centroid)

        for i in range(k):
            pts=data[labels==i]
            if len(pts)>0:
                new_centroid[i]=pts.mean(axis=0)
            else:
                new_centroid[i]=centroid[i]

        s=np.linalg.norm(new_centroid-centroid)
        centroid=new_centroid

        if s<tol:
            break

    pool.close()
    pool.join()
    total=time.perf_counter()-start
    return total,process

def min_points(data,k):
    n,d=data.shape
    work=max(k*d,1)
    target=int(0.5e6*CPU_COUNT)

    est=int(target/work)
    est=max(500,est)
    est=min(n//2 if n>1 else 1,est)

    return est

def min_process(n,m,cpu_count):
    if n<=m:
        return 1
    
    est=n//m
    process=min(cpu_count,est)

    if n/process<m:
        process=max(1,n//m)

    return max(1,process)

def split_data(data,p):
    chunk=int(np.ceil(len(data)/p))

    return [data[i:i+chunk] for i in range(0,len(data),chunk)]

def DLBP(data,k):
    n,d=data.shape
    M_P=min_points(data,k)
    proc=min_process(n,M_P,CPU_COUNT)

    centroids=data[:k].copy()
    chunks=split_data(data,proc)

    pool=Pool(proc)
    start=time.perf_counter()

    for _ in range(max_iter):
        results=pool.map(assign_chunk,[(c,centroids) for c in chunks])

        labels=np.concatenate(results)
        new_centroids=np.zeros_like(centroids)

        for i in range(k):
            pts=data[labels==i]

            if len(pts)>0:
                new_centroids[i]=pts.mean(axis=0)
            else:
                new_centroids[i]=centroids[i]

        s=np.linalg.norm(new_centroids-centroids)
        centroids=new_centroids

        if s<tol:
            break

    pool.close()
    pool.join()
    total=time.perf_counter()-start
    return total,proc

def init_worker(shm_name,shape,dtype):
    global SHM_NAME,SHM_SHAPE,SHM_DTYPE

    SHM_NAME=shm_name
    SHM_SHAPE=shape
    SHM_DTYPE=dtype

def assign_chunk_shm(args):
    start,end,centroid=args
    shm=shared_memory.SharedMemory(name=SHM_NAME)
    data=np.ndarray(SHM_SHAPE,dtype=SHM_DTYPE,buffer=shm.buf)

    chunk=data[start:end]
    dist=np.linalg.norm(chunk[:,None]-centroid,axis=2)
    label=np.argmin(dist,axis=1)

    return start,end,label

def DLBSMP(data,k,processes):
    n,d=data.shape
    shm=shared_memory.SharedMemory(create=True,size=data.nbytes)
    shm_array=np.ndarray(data.shape,dtype=data.dtype,buffer=shm.buf)
    shm_array[:]=data[:]

    chunk_size=int(np.ceil(n/processes))
    ranges=[(i,min(i+chunk_size,n)) for i in range(0,n,chunk_size)]

    centroids=data[:k].copy()

    pool=Pool(processes,initializer=init_worker,initargs=(shm.name,data.shape,data.dtype))

    labels=np.empty(n,dtype=np.int32)
    start=time.perf_counter()

    for _ in range(max_iter):
        results=pool.map(assign_chunk_shm,[(s,e,centroids) for s,e in ranges])

        for s,e,l in results:
            labels[s:e]=l

        new_centroids=np.zeros_like(centroids)
        counts=np.bincount(labels,minlength=k)

        for dim in range(d):new_centroids[:,dim]=np.bincount(labels,weights=data[:,dim],minlength=k)

        nonzero=counts>0
        new_centroids[nonzero]/=counts[nonzero,None]
        new_centroids[~nonzero]=centroids[~nonzero]
        shift=np.linalg.norm(new_centroids-centroids)

        centroids=new_centroids
        if shift<tol:
            break

    pool.close()
    pool.join()

    shm.close()
    shm.unlink()

    total=time.perf_counter()-start
    return total,chunk_size

def experiment(data,k,outfile):
    seq_time=SE(data,k)
    outfile.write(f"Sequential Time: {seq_time:.4f}\n\n")

    old_time,old_proc=ECP(data,k)
    old_speedup=seq_time/old_time
    outfile.write("ECP method\n")
    outfile.write(f"Processors Used: {old_proc}\n")
    outfile.write(f"Execution Time: {old_time:.4f}\n")
    outfile.write(f"Speedup: {old_speedup:.4f}\n\n")

    adap1_time,adap1_proc=DLBP(data,k)
    adap1_speedup=seq_time/adap1_time
    alpha1=(1/adap1_speedup - 1/adap1_proc)/(1 - 1/adap1_proc)
    outfile.write("DLBP method \n")
    outfile.write(f"Processors Used: {adap1_proc}\n")
    outfile.write(f"Execution Time: {adap1_time:.4f}\n")
    outfile.write(f"Speedup: {adap1_speedup:.4f}\n")
    outfile.write(f"Amdahl's Law: Alpha= {alpha1:.6f}\n")

    times=[]
    chunks=[]
    procs=[]
    
    outfile.write("\nDLBSMP method\n")
    for p in Test_p:
        if p>CPU_COUNT:
            continue

        print("Testing processors:",p)
        t,chunk=DLBSMP(data,k,p)
        times.append(t)
        chunks.append(chunk)
        procs.append(p)

        speedup=seq_time/t
        outfile.write(f"Processors {p} | Time {t:.4f} | Speedup {speedup:.4f} | Chunk {chunk}\n")
    
    optimal_i=np.argmin(times)
    optimal_p=  procs[optimal_i]
    optimal_time =times[optimal_i]
    adaptive_speedup=seq_time/optimal_time
    alpha=(1/adaptive_speedup -1/optimal_p)/(1-1/optimal_p)
    outfile.write("\nOPTIMAL SHARED MEMORY CONFIGURATION\n")
    outfile.write(f"Optimal Processors: {optimal_p}\n")
    outfile.write(f"Optimal Time: {optimal_time:.4f}\n")
    outfile.write(f"Speedup: {adaptive_speedup:.4f}\n")
    outfile.write(f"Amdahl's Law: Alpha= {alpha:.6f}\n")
    
    return seq_time,old_speedup,adap1_speedup,alpha1,procs,times,adaptive_speedup,alpha

def bar_graph(dataset,seq_time,old_speedup,adap1_speedup,adap2_speedup):
    name=dataset.split(".")[0]
    label=["ECP","DLBP","DLBSMP"]

    val=[old_speedup,adap1_speedup,adap2_speedup]

    plt.figure()
    plt.bar(label,val)
    plt.title("Speedup Comparison of Parallel k-Means Variants")
    plt.xticks(rotation=10)
    plt.ylabel("Speed up")
    plt.grid(axis='y')
    plt.savefig(name+"_overall_comparison.png")
    plt.show()

def r_plots(dataset,seq_time,procs,time,alpha):
    ad_speedup = [seq_time/t for t in time]
    amdahl_speedup=[1/(alpha+(1-alpha)/p) for p in procs]
    name=dataset.split(".")[0]

    plt.figure()
    plt.plot(procs,time,marker='o',color="orange",label="DLBSMP")
    plt.xlabel("Processors")
    plt.ylabel("Execution Time")
    plt.title("Processor vs Execution Time")
    plt.grid()
    plt.legend()
    plt.savefig(name+"_processor_vs_time.png")
    plt.show()

    plt.figure()
    plt.plot(procs,ad_speedup,marker='o',color="red",label="DLBSMP")
    plt.xlabel("Processors")
    plt.ylabel("Speedup")
    plt.title("Processor vs Speedup")
    plt.legend()
    plt.grid()
    plt.savefig(name+"_processor_vs_speedup.png")
    plt.show()

    plt.figure()
    plt.plot(procs,ad_speedup,marker='o',color='orange',label="DLBSMP")
    plt.plot(procs,amdahl_speedup,marker='d',color='blue',label="Amdahl")
    plt.xlabel("Processors")
    plt.ylabel("Speedup")
    plt.title("DLBSMP vs Amdahl")
    plt.legend()
    plt.savefig(name+"shared_memory vs amdahl.png")
    plt.grid()
    plt.show()

def main():
    filename=input("Enter dataset file: ")
    k=int(input("Enter number of clusters: "))
    data=load_dataset(filename)
    output=get_output_filename(filename)

    f=open(output,"w")
    f.write("Dataset: "+filename+"\n")
    f.write(f"Points: {data.shape[0]}\n")
    f.write(f"Dimensions: {data.shape[1]}\n\n")
    seq_time,old_speedup,adap1_speedup,alpha1,procs,times,adap2_speedup,alpha=experiment(data,k,f)
    f.close()

    print("Results written to",output)
    r_plots(filename,seq_time,procs,times,alpha)
    bar_graph(filename,seq_time,old_speedup,adap1_speedup,adap2_speedup)

if __name__=="__main__":
    main()
