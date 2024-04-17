import random
import numpy as np
import pygame
from pygame import gfxdraw
import pyspiel
from open_spiel.python.algorithms import mcts
from othello_game import OthelloGame, OthelloState

pygame.init()
pygame.font.init()

myfont = pygame.font.SysFont("Comis Sans MS", 80)

DEFAULT_N = 4

pix = 80
half = pix // 2
linew = 1
winw = (pix + linew) * DEFAULT_N - linew
background = (0, 144, 103)
black = (0, 0, 0)
grey = (48, 48, 48)
white = (255, 255, 255)
red = (255, 0, 0)

display_width = winw
display_height = winw

win = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Reversi")


def draw_game_board(board, mboard):
    win.fill(background)

    for x in range(1, DEFAULT_N):
        pygame.draw.rect(win, black, (x * (pix + linew), 0, linew, winw))
    for y in range(1, DEFAULT_N):
        pygame.draw.rect(win, black, (0, y * (pix + linew), winw, linew))

    for x in range(DEFAULT_N):
        for y in range(DEFAULT_N):
            if board[x][y] == 1:
                gfxdraw.filled_circle(
                    win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, black
                )
                gfxdraw.aacircle(
                    win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, black
                )
            elif board[x][y] == -1:
                gfxdraw.filled_circle(
                    win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, white
                )
                gfxdraw.aacircle(
                    win, x * (pix + linew) + half, y * (pix + linew) + half, half - 6, black
                )

    for x in range(DEFAULT_N):
        for y in range(DEFAULT_N):
            if mboard[x][y]:
                gfxdraw.aacircle(
                    win,
                    x * (pix + linew) + half,
                    y * (pix + linew) + half,
                    half - 6,
                    grey,
                )


if __name__ == "__main__":
    game = pyspiel.load_game("othello_nxn")
    state = game.new_initial_state()

    # player1 is black, player2 is white
    player1 = "player"
    # player2 = "player"
    # mcts ai
    rng = np.random.RandomState(42)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    player2 = mcts.MCTSBot(
        game, uct_c=2, max_simulations=1000, evaluator=evaluator, random_state=rng
    )

    clock = pygame.time.Clock()
    run = True
    printed_winner = False

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
                        action = my + mx * DEFAULT_N
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
                        action = my + mx * DEFAULT_N
                        if action in state.legal_actions():
                            state.apply_action(action)
                else:
                    action = player2.step(state)
                    state.apply_action(action)

        if state.is_terminal() and not printed_winner:
            printed_winner = True
            print("Winner: ", state.returns()[0])

        board = (
            np.array(state.observation_tensor(0)).reshape(2, DEFAULT_N, DEFAULT_N)[0]
            - np.array(state.observation_tensor(0)).reshape(2, DEFAULT_N, DEFAULT_N)[1]
        )
        mboard = np.zeros(DEFAULT_N * DEFAULT_N, dtype=np.int8)
        mboard[state.legal_actions()] = 1
        draw_game_board(board, mboard.reshape(DEFAULT_N, DEFAULT_N))

        if state.is_terminal():
            if state.returns()[0] == 1:
                text = myfont.render("Black Wins!", True, red)
            elif state.returns()[0] == -1:
                text = myfont.render("White Wins!", True, red)
            else:
                text = myfont.render("Draw!", True, red)
            win.blit(
                text,
                (
                    display_width // 2 - text.get_width() // 2,
                    display_height // 2 - text.get_height() // 2,
                ),
            )

        pygame.display.update()

        clock.tick(20)

    pygame.quit()
