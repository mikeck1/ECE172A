import matplotlib.pyplot as plt
import numpy as np
# im = pickle.read('172maze2021.p')
import pandas as pd

def notEdge(pos):
    if -1 in pos or 50 in pos: return False
    return True
def isSame(pos1,pos2):
    if pos1[0] is pos2[0] and pos1[1] is pos2[1]: return True
    return False

def bfs(start,end, im):
    path = []
    traversed  = []
    q = [[start]]
    while q:
        path = q.pop()
        start = path[-1]
        traversed.append(start)
        if isSame(start,end): return (path, traversed)
        
        theDirN = (start[0],start[1]+1)
        theDirE = (start[0]+1,start[1])
        theDirS = (start[0],start[1]-1)
        theDirW = (start[0]-1,start[1])
        
#         print(start)
        
        if notEdge(theDirN):
#             print("hi")
            if im[start][0]:
#                 print(theDirN,im[start][0])
                im[start][0]=False
                new_path = path.copy()
                new_path.append(theDirN)
                q.append(new_path)
                
        if im[start][1]:
            im[start][1]=False
            if notEdge(theDirE): 
                new_path = path.copy()
                new_path.append(theDirE)
                q.append(new_path)
                
        if im[start][2]:
            im[start][2]=False
            if notEdge(theDirS): 
                new_path = path.copy()
                new_path.append(theDirS)
                q.append(new_path)
                
        if im[start][3]:
            im[start][3]=False
            if notEdge(theDirW): 
                new_path = path.copy()
                new_path.append(theDirW)
                q.append(new_path)
                
                
def draw_path(final_path_points, other_path_points):
    '''
    final_path_points: the list of points (as tuples or lists) comprising your final maze path. 
    other_path_points: the list of points (as tuples or lists) comprising all other explored maze points. 
    (0,0) is the start, and (49,49) is the goal.
    Note: the maze template must be in the same folder as this script.
    '''
    im = plt.imread('172maze2021.png')
    x_interval = (686-133)/49
    y_interval = (671-122)/49
    plt.imshow(im)
    fig = plt.gcf()
    ax = fig.gca()
    circle_start = plt.Circle((133, 800-122), radius=4, color='lime')
    circle_end = plt.Circle((686, 800-671), radius=4, color='red')
    ax.add_patch(circle_start)
    ax.add_patch(circle_end)
    for point in other_path_points:
        if not (point[0] == 0 and point[1] == 0) and not (point[0] == 49 and point[1] == 49):
            circle_temp = plt.Circle(
                (133+point[0]*x_interval, 800-(122+point[1]*y_interval)), radius=4, color='blue')
            ax.add_patch(circle_temp)
    for point in final_path_points:
        if not (point[0] == 0 and point[1] == 0) and not (point[0] == 49 and point[1] == 49):
            circle_temp = plt.Circle(
                (133+point[0]*x_interval, 800-(122+point[1]*y_interval)), radius=4, color='yellow')
            ax.add_patch(circle_temp)
    plt.show()
    
    
pic = pd.read_pickle('172maze2021.p')
thing = bfs((0,0),(49,49),pic)

draw_path(thing[0],thing[1])