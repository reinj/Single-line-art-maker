# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:01:03 2019

@author: Rauno
"""

import cv2
import numpy as np
from time import clock

def findClosest(c1,c2):
    c1 = np.squeeze(c1, axis=1)
    c2 = np.squeeze(c2, axis=1)
    pfc1 = None
    pfc2 = None
    cd = 1000
    for p1 in c1:
        for p2 in c2:
            ppd = np.linalg.norm(p2-p1)
            if ppd < cd :
                cd = ppd
                pfc1 = p1
                pfc2 = p2
    rp1 = tuple((int(pfc1[0]), int(pfc1[1])))
    rp2 = tuple((int(pfc2[0]), int(pfc2[1])))
    return rp1, rp2, cd

def show(img, name):    # kuvamine
    y, x = img.shape
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, x, y)
    cv2.imshow(name,img)

#img = cv2.imread("./index.png", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("./index2.png", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("./R3.png", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("./1547769148.png", cv2.IMREAD_GRAYSCALE) # Kirvepilt - ilma grayscaleta tuleks vbla parem
t0 = clock()
img = cv2.imread("./Frankenstein.jpg", cv2.IMREAD_GRAYSCALE)
ret,thresh = cv2.threshold(img,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
t1 = clock()
#cv2.imwrite('contours.png',contours)
cim = np.zeros_like(img)
#cv2.line(cim, tuple((0,0)), tuple((100,100)), 255, 1)
contours = [c for c in contours if len(c)>20] #hästi väikestest kontuuridest on savi

contours = [c for c in contours if not np.all(c[1] == c[-1])] # kontuuridest mis on jooned on savi
t2 = clock()

graph = [[c] for c in contours] # viskame iga kontuuri omaette basketisse
L = len(contours)
dists = np.full((L,L),9999)
locs = np.full((L,L,2),np.zeros((1,2)),dtype=np.uint32)
print("Total",L,"contours")
for i in range(L):
    for j in range(i+1,L):
        print("Calculating distances between",i,"and",j)
        p1, p2, dist = findClosest(contours[i],contours[j])
        dists[i,j] = dist
        dists[j,i] = dist
        locs[i,j] = p1
        locs[j,i] = p2
t3 = clock()
cv2.drawContours(cim, contours, -1, 255, 1)

connectPairs = []        
while (len(graph) > 1) :
    #_graph = []
    _from = -1
    _to = -1
    _distance = 9999
    for contr in graph[0] :
        #index = contours.index(contr)
        index = [idx for idx, el in enumerate(contours) if np.array_equal(el, contr)][0]
        #print("Checking to connect",index)
        while True :
            close = min(dists[index])
            if close < _distance :
                nodeNr = np.where(dists[index] == close)[0][0]
                if len([idx for idx, el in enumerate(graph[0]) if np.array_equal(el, contours[nodeNr])]) == 0 :
                    #if not np.isin(contours[nodeNr], graph[0]).any():
                    #if contours[nodeNr] not in graph[0] :
                    _from = index
                    _to = nodeNr
                    _distance = close
                    break
                else :
                    dists[index,nodeNr] = 9999
                    dists[nodeNr,index] = 9999
            else :
                break
    
    # find which graph index contains contours[nodeNr]
    _graph = []
    print("Want to connect",_from,"to",_to)
    for i in range(1,len(graph)):
        if len([idx for idx, el in enumerate(graph[i]) if np.array_equal(el, contours[_to])]) > 0:
        #if contours[_to] in graph[i] :
            _graph = graph[i]
            break
    else :
        print("Bad")
        break
    graph[0] = graph[0] + _graph
    graph = graph[:i] + graph[i+1:]
    
    dists[_from,_to] = 9999
    dists[_to,_from] = 9999
    p1 = tuple(locs[_from,_to])
    p2 = tuple(locs[_to,_from])
    p1id = [idx for idx, el in enumerate(np.squeeze(contours[_from], axis=1)) if np.array_equal(el, p1)][0]
    p2id = [idx for idx, el in enumerate(np.squeeze(contours[_to], axis=1)) if np.array_equal(el, p2)][0]
    # To avoid indexerror
    p1id = p1id if p1id+1 < len(contours[_from]) else -1
    p2id = p2id if p2id+1 < len(contours[_to]) else -1
    
    _p11x = contours[_from][p1id-1][0][0]
    _p11y = contours[_from][p1id-1][0][1]
    _p12x = contours[_from][p1id+1][0][0]
    _p12y = contours[_from][p1id+1][0][1]
    _p21x = contours[_to][p2id-1][0][0]
    _p21y = contours[_to][p2id-1][0][1]
    _p22x = contours[_to][p2id+1][0][0]
    _p22y = contours[_to][p2id+1][0][1]
    
    cv2.line(cim,(_p11x,_p11y),(_p22x,_p22y), 255, 1)
    cv2.line(cim,(_p12x,_p12y),(_p21x,_p21y), 255, 1)
    
    cim[p1[::-1]] = 0
    cim[p2[::-1]] = 0
    
    #print(np.linalg.norm((_p22x-_p21x,_p22y-_p21y)))
    #print(np.linalg.norm((_p12x-_p11x,_p12y-_p11y)))
    connectPairs.append((p1,p2))
    
t4 = clock()
print(dists)
print(connectPairs)

show(cim,"1234")
print("Finding contours:",t1-t0)
print("Preprocessing:",t2-t1)
print("Algorithm:",t3-t2)
print("Connecting nearest:",t4-t3)
#print(len(contours))
#contours = np.concatenate(contours)
#cv2.drawContours(cim, contours, -1, 255, 1)

#Teeb Negatiivi
#print('generating negative')
#imagem = cv2.bitwise_not(cim)
#cv2.imshow('inverted', imagem)



"""
print('powering up the algortithm')
#Alates siit on enamus netist kopitud, tuleb ära kaotada
height, width = img.shape
bw_image_array = np.array(imagem, dtype=np.int)
black_indices = np.argwhere(bw_image_array == 0)
chosen_black_indices = black_indices#[np.random.choice(black_indices.shape[0], replace=False, size=100)]

distances = pdist(chosen_black_indices)
distance_matrix = squareform(distances)
print('starting TSP')
optimized_path = solve_tsp(distance_matrix)
print('starting TSP points')


optimized_path_points = [chosen_black_indices[x] for x in optimized_path]
lined = np.zeros_like(img)
#s = optimized_path_points[0]
for e in optimized_path_points:
    try :
    #e = optimized_path_points[x]
        lined[e[0],e[1]] = 255
    except IndexError :
        pass
    #cv2.line(lined, tuple((s[1],s[0])), tuple((e[1],e[0])), 255, 1)
    #s = e
show(lined, "Hello")"""
#show(imagem,"Contours")
#plt.figure(figsize=(8, 10), dpi=100)
#plt.plot([x[1] for x in optimized_path_points], [x[0] for x in optimized_path_points], color='black', lw=1)
#plt.xlim(0, width)
#plt.ylim(0, height)
#plt.gca().invert_yaxis()
#plt.xticks([])
#plt.yticks([])
#plt.savefig('filtered.png', bbox_inches='tight')
print('Image ready and saved')