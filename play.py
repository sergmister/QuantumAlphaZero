import random
import numpy as np
import pygame
from pygame import gfxdraw
import pyspiel
from open_spiel.python.algorithms import mcts

pygame.init()
pygame.font.init()

myfont = pygame.font.SysFont('Comis Sans MS', 80)
game_end_font = pygame.font.SysFont('Comis Sans MS', 200)

pix = 80
half = pix // 2
linew = 1
winw = (pix + linew) * 8 - linew
background = (0, 144, 103)
black = (0, 0, 0)
grey = (48, 48, 48)
white = (255, 255, 255)

display_width = winw
display_height = winw

win = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Reversi")


def redrawGameWindow(board, mboard):
    win.fill(background)

    for x in range(1, 8):
        pygame.draw.rect(win, black, (x * (pix + linew), 0, linew, winw))
    for y in range(1, 8):
        pygame.draw.rect(win, black, (0, y * (pix + linew), winw, linew))

    for x in range(8):
        for y in range(8):
            if board[x][y] == 1:
                gfxdraw.filled_circle(win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, black)
                gfxdraw.aacircle(win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, black)
            elif board[x][y] == -1:
                gfxdraw.filled_circle(win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, white)
                gfxdraw.aacircle(win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, black)

    for x in range(8):
        for y in range(8):
            if mboard[x][y]:
                gfxdraw.aacircle(win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, grey)

    pygame.display.update()


if __name__ == "__main__":
    game = pyspiel.load_game("othello")
    state = game.new_initial_state()

    # player1 is black, player2 is white
    player1 = "player"
    # player2 = "player"
    # mcts ai
    rng = np.random.RandomState(42)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    player2 = mcts.MCTSBot(game, uct_c=2, max_simulations=1000, evaluator=evaluator, random_state=rng)

    clock = pygame.time.Clock()
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        if not state.is_terminal():
            if state.current_player() == 0:
                if player1 == "player":
                    if pygame.mouse.get_pressed()[0]:
                        mx, my = pygame.mouse.get_pos()
                        mx = mx // (pix + linew)
                        my = my // (pix + linew)
                        action = my + mx * 8
                        if action in state.legal_actions():
                            state.apply_action(action)
                else:
                    action = player2.step(state)
                    state.apply_action(action)

            elif state.current_player() == 1:
                if player2 == "player":
                    if pygame.mouse.get_pressed()[0]:
                        mx, my = pygame.mouse.get_pos()
                        mx = mx // (pix + linew)
                        my = my // (pix + linew)
                        action = my + mx * 8
                        if action in state.legal_actions():
                            state.apply_action(action)
                else:
                    action = player2.step(state)
                    state.apply_action(action)

        clock.tick(20)
        board = np.array(state.observation_tensor(0)).reshape(3, 8, 8)[1] - np.array(state.observation_tensor(0)).reshape(3, 8, 8)[2]
        mboard = np.zeros(64, dtype=np.int8)
        if not state.is_terminal():
            mboard[state.legal_actions()] = 1
        redrawGameWindow(board, mboard.reshape(8, 8))

    pygame.quit()