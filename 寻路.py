#导入库
import random
import sys
import pygame
#创建三个列表
to_be_selected = []
random_selectB = []
path_list = []


#设置长宽
ROWS = 30
COLUMNS = 30

#生成地图
x = [0, 2, 0, -2]
y = [2, 0, -2, 0]

#寻路的四方寻路
px = [0, 1, 0, -1]
py = [1, 0, -1, 0]


#列表套列表，全是0
#i*j阶列表
#list = [[0,0,0],[0,0,0],[0,0,0]]
#得到路
isvisit = [[0 for i in range(COLUMNS)] for j in range(ROWS)]
isvisit[1][1] = 1

#传入参数r,c
def matrix_init(r, c):
    #创建一个列表套列表，全是1
    matrix = [[1 for i in range(c)] for j in range(r)]
    matrix[1][1] = 0
    return matrix

# start = [1, 1]
# put_node_in_to_be_selected(start)
def put_node_in_to_be_selected(node):
    for i in range(4):
        #列举四方
        xx = node[0] + x[i]
        yy = node[1] + y[i]

        # x = [0, 2, 0, -2]
        # y = [2, 0, -2, 0]

        # W D S A

        #xx = 1,yy = 3
        #xx = 3,yy = 1
        #xx = 1,yy = -1
        #xx = -1,yy = 1

        #添加到墙
        if xx > 0 and xx < ROWS and yy > 0 and yy < COLUMNS and ([xx, yy] not in to_be_selected) and matrix[xx][
            yy] == 1:
            to_be_selected.append([xx, yy])

#得到墙
matrix = matrix_init(ROWS, COLUMNS)


def random_B(node):
    #清除列表random_selectB
    random_selectB.clear()
    #四方
    for i in range(4):

        xx = node[0] + x[i]
        yy = node[1] + y[i]

        #列表添加到路
        if xx > 0 and xx < ROWS and yy > 0 and yy < COLUMNS and matrix[xx][yy] == 0:
            random_selectB.append([xx, yy])
    rand_B = random.randint(0, len(random_selectB) - 1)
    return random_selectB[rand_B]

#开始坐标
start = [1, 1]
#添加方位坐标
put_node_in_to_be_selected(start)
#重置开始坐标
path_list.append([1, 1])


def matrix_generate():
    if len(to_be_selected) > 0:
        #随机数生成
        rand_s = random.randint(0, len(to_be_selected) - 1)
        #得到在  to_be_selected  列表中的值
        select_nodeA = to_be_selected[rand_s]
        # 得到在 random_selectB  列表中的值
        selectB = random_B(select_nodeA)
        #
        matrix[select_nodeA[0]][select_nodeA[1]] = 0

        mid_x = int((select_nodeA[0] + selectB[0]) / 2)
        mid_y = int((select_nodeA[1] + selectB[1]) / 2)
        matrix[mid_x][mid_y] = 0
        put_node_in_to_be_selected(select_nodeA)
        to_be_selected.remove(select_nodeA)
    elif len(path_list) > 0:
        matrix[ROWS - 2][COLUMNS - 2] = 3
        l = len(path_list) - 1
        n = path_list[l]
        if n[0] == ROWS - 2 and n[1] == COLUMNS - 2:
            return
        for i in range(4):
            xx = n[0] + px[i]
            yy = n[1] + py[i]
            if xx > 0 and xx < ROWS - 1 and yy > 0 and yy < COLUMNS - 1 and (
                    matrix[xx][yy] == 0 or matrix[xx][yy] == 3) and isvisit[xx][yy] == 0:
                isvisit[xx][yy] = 1
                matrix[n[0]][n[1]] = 2
                tmp = [xx, yy]
                path_list.append(tmp)
                break
            elif i == 3:
                matrix[n[0]][n[1]] = 0
                path_list.pop()


pygame.init()
screen = pygame.display.set_mode((COLUMNS * 15, ROWS * 15))
pygame.display.set_caption("生成迷宫地图")


def draw_rect(x, y, color):
    pygame.draw.rect(screen, color, ((y * 15, x * 15, 15, 15)))


def draw_maze():
    for i in range(ROWS):
        for j in range(COLUMNS):
            if matrix[i][j] == 1:
                draw_rect(i, j, "blue")
            if matrix[i][j] == 2:
                draw_rect(i, j, "green")
            if matrix[i][j] == 3:
                draw_rect(i, j, "red")


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill("white")
    matrix_generate()
    draw_maze()
    pygame.display.flip()
    pygame.time.Clock().tick(30)