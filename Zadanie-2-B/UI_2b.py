import tkinter as tk
import numpy as np
import random
import time

root=tk.Tk()
max_x_can,max_y_can=600,600;
canvas=tk.Canvas(root, width=max_x_can, height=max_y_can);
canvas.pack()

canvas.create_rectangle(50,50,550,550);

MAX_X, MAX_Y=5000,5000
MIN_X, MIN_Y=-5000,-5000

array_ran_sur=[]
num_points=10000
num_clus=20
radius=1
scaling_down=20
distances=np.zeros((num_points+20, num_clus))


def init_20(array,radius,scale):
    global MAX_X,MAX_Y,MIN_X,MIN_Y
    for i in range(20):
        ran_x=random.randrange(MIN_X,MAX_X)
        ran_y=random.randrange(MIN_Y,MAX_Y)
        ran_sur=[ran_x,ran_y]
        array.append(ran_sur)

        coord_x=(ran_x//scale)+250+50   #+50 kvoli okraju
        coord_y=(ran_y//scale)+250+50
        canvas.create_oval(coord_x-radius,coord_y-radius,coord_x+radius,coord_y+radius, fill='black')
    #print(array_ran_sur)


def generate_more(arr, count,radius,scale):
    point=random.choice(arr)

    #print(f"Generovanie cisla c.{count}")
    #print(point)

    #zredukovat interval!!!!
    X_offset=random.randint(-100,100)
    Y_offset=random.randint(-100,100)
    
    while point[0]+X_offset>5000 or point[0]+X_offset<-5000 or point[1]+Y_offset>5000 or point[1]+Y_offset<-5000:
        if point[0]+X_offset>5000 or point[0]+X_offset<-5000:
            X_offset=random.randint(-100,100)
        elif point[1]+Y_offset>5000 or point[1]+Y_offset<-5000:
            Y_offset=random.randint(-100,100)
        else:
            continue

    new_x=point[0]+X_offset
    new_y=point[1]+Y_offset
    new=[new_x,new_y]

    arr.append(new)

    coord_x=(new[0]//scale)+250+50   #+50 kvoli okraju
    coord_y=(new[1]//scale)+250+50

    canvas.create_oval(coord_x-radius,coord_y-radius,coord_x+radius,coord_y+radius, fill="black", outline="")
    return new


def kmeans_centroid():
    pass

#vyber medoidov nahodne
def kmeans_medoid(arr, k, radius, scale,medoids_id):
    medoids_sample=random.sample(arr,k)
    for num in range(k):
        medoids_id.append(medoids_sample[num])

    #print(f"zoznam medoidov: {medoids_id}")

    for i in range(k):
        coord_x=(medoids_sample[i][0]//scale)+250+50   #+50 kvoli okraju
        coord_y=(medoids_sample[i][1]//scale)+250+50
        canvas.create_oval(coord_x-radius,coord_y-radius,coord_x+radius,coord_y+radius, width=5, outline="#24fc03")

#vypocet vzdialenosti bodov od medoidov, distances => maxtrix POINTS x MEDOIDS
def dist_calc(distances, array_m, array_points):
    #print(len(array_points))
    #print(len(array_m))

    for i in range(len(array_points)):
        for j in range(len(array_m)):
            medoid=np.array([array_m[j][0],array_m[j][1]])
            point=np.array([array_points[i][0],array_points[i][1]])
            distances[i,j]=round(np.linalg.norm(medoid-point),2)

#priradenie bodov ku klasterom
def clustering(distances, clusters, array):
    prirad=np.argmin(distances, axis=1)                     #axis=1 => prvy stlpec matrixu
    priradenie_zoznam=prirad.tolist()                       #premena na pouzitelne pole
    for point_index, cluster_index in enumerate(priradenie_zoznam):
        clusters[cluster_index].append(array[point_index])  #priradenie specifickeho bodu (nie index bodu) k polu specifickeho indexu klastera

def update_med(new_medoids, clusters):
    print("\nupdate funkcia")
    radius=1
    for cluster in clusters:
        distances_in_cluster=[]
        for point in cluster:
            total_sum=sum(np.linalg.norm((np.array(point))-np.array(other_point)) for other_point in cluster)
            distances_in_cluster.append((point, total_sum))

            new_medoid=min(distances_in_cluster, key=lambda x: x[1])[0]
            new_medoids.append(new_medoid)
            #print(f"{new_medoids}")

        coord_x=(new_medoid[0]//20)+250+50   #+50 kvoli okraju
        coord_y=(new_medoid[1]//20)+250+50
        canvas.create_oval(coord_x-radius, coord_y-radius, coord_x+radius, coord_y+radius, width=5, outline="orange")
    
    print("update done")
    return new_medoids

def div_cluster():
    pass


start=time.time()
#inicializacia prvych 20 bodov a dodatocnych 40k bodov
print("inic")
init_20(array_ran_sur,radius,scaling_down);
for count in range(num_points):
    #print(generate_more(array_ran_sur, count+1,radius,scaling_down))
    generate_more(array_ran_sur, count+1,radius,scaling_down)
num_points+=20

#-----------------------------------------------
#moznost k-means s centrom medoid
medoids_id=[]
new_medoids=[]

#inicializacia prazdnych poli pre klastery
clusters=[[] for _ in range(num_clus)]
kmeans_medoid(array_ran_sur,num_clus,radius,scaling_down,medoids_id)
dist_calc(distances, medoids_id, array_ran_sur)
clustering(distances,clusters,array_ran_sur)
update_med(new_medoids,clusters)
#-----------------------------------------------
end=time.time()
print(end-start)

root.mainloop();
