# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue
from queue import PriorityQueue, Queue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    frontier = Queue()
    explored = dict()
    frontier.put(maze.start)
    
    #if not a wall and not the start and not already in the frontier
    #else add into explored
    while frontier.qsize() > 0:
        x,y = frontier.get()
        if (x,y) == maze.waypoints[0]: 
            break

        if (maze[x, y-1] != maze.legend.wall) and (maze[x,y-1] != maze.legend.start) and ((x,y-1) not in explored):
            frontier.put((x, y-1))
            explored[(x,y-1)] = (x,y)

        if (maze[x, y+1] != maze.legend.wall) and (maze[x,y+1] != maze.legend.start) and ((x,y+1) not in explored):
            frontier.put((x, y+1))
            explored[(x,y+1)] = (x,y)

        if (maze[x-1, y] != maze.legend.wall) and (maze[x-1,y] != maze.legend.start) and ((x-1,y) not in explored):
            frontier.put((x-1, y))
            explored[(x-1,y)] = (x,y)

        if (maze[x+1, y] != maze.legend.wall) and (maze[x+1,y] != maze.legend.start) and ((x+1,y) not in explored):
            frontier.put((x+1, y))
            explored[(x+1,y)] = (x,y)

    out = Queue()
    front = maze.waypoints[0]
    out.put(front)
    while front != maze.start:
        front = explored[front]
        out.put(front)
    
    li = []
    while out.qsize() > 0:
        li.append(out.get())
    
    while out.qsize() > 0:
       li.append(out.get())
    li.reverse()   
    
    return li

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    q = PriorityQueue()
    back = dict()
    #backSet = set()
    endX, endY = maze.waypoints[0]
    q.put((abs(maze.start[0]-endX) + abs(maze.start[1]-endY), maze.start))
    while q.qsize() > 0:
        _, b = q.get()
        x,y = b
        if (x,y) == maze.waypoints[0]:
            break
        
        pos = b
        count = 0
        while pos != maze.start:
            pos = back[pos]
            #pos = backSet.pop()
            count += 1

        if maze[x, y-1] != maze.legend.wall and maze[x,y-1] != maze.legend.start and (x,y-1) not in back:
            q.put((count + abs(x-endX) + abs((y-1)-endY), (x,y-1)))
            back[(x,y-1)] = (x,y)
            #backSet.add((x,y-1))

        if maze[x, y+1] != maze.legend.wall and maze[x,y+1] != maze.legend.start and (x,y+1) not in back:
            q.put((count + abs(x-endX) + abs((y+1)-endY), (x,y+1)))
            back[(x,y+1)] = (x,y)
            #backSet.add((x,y+1))

        if maze[x-1, y] != maze.legend.wall and maze[x-1,y] != maze.legend.start and (x-1,y) not in back:
            q.put((count + abs((x-1)-endX) + abs(y-endY), (x-1,y)))
            back[(x-1,y)] = (x,y)
            #backSet.add((x-1,y))

        if maze[x+1, y] != maze.legend.wall and maze[x+1,y] != maze.legend.start and (x+1,y) not in back:
            q.put((count + abs((x+1)-endX) + abs(y-endY), (x+1,y)))
            back[(x+1,y)] = (x,y)
            #backSet.add((x+1,y))

    out = Queue()
    end = maze.waypoints[0]
    out.put((end))
    while end != maze.start:
        end = back[end]
        #end = backSet.pop()
        out.put(end)
    li = []
    while out.qsize() > 0:
       li.append(out.get())
    
    li.reverse()   
    return li


# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
