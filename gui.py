# Third party
import pygame
import easygui
import numpy as np

# My own module
# import Neural

pygame.init()

pygame.display.set_caption("Angel-IG: Handwritten digits guesser")
WIDTH, HEIGHT = 700, 700

dimCW = WIDTH / 28
dimCH = HEIGHT / 28

screen = pygame.display.set_mode((HEIGHT, WIDTH))

BG = 25, 25, 25

exit_gui = False
print_guessed_number = False
current_image = np.zeros((28, 28))

while not exit_gui:
    screen.fill(BG)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_gui = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                exit_gui = True
            elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                print_guessed_number = True

    mouse_click = pygame.mouse.get_pressed()

    # If there's a left-side or right-side click:
    if mouse_click[0] or mouse_click[2]:
        posX, posY = pygame.mouse.get_pos()
        pixelX, pixelY = int(np.floor(posX / dimCW)), int(np.floor(posY / dimCH))
        current_image[pixelX, pixelY] = 1

    for y in range(0, 28):
        for x in range(0, 28):
            poly = [
                (int(x * dimCW), int(y * dimCH)),
                (int((x + 1) * dimCW), int(y * dimCH)),
                (int((x + 1) * dimCW), int((y + 1) * dimCH)),
                (int(x * dimCW), int((y + 1) * dimCH)),
            ]

            if not current_image[x, y]:
                pygame.draw.polygon(screen, (128, 128, 128), poly, 1)
            else:
                pygame.draw.polygon(screen, (255, 255, 255), poly, 0)

    if print_guessed_number:
        easygui.msgbox("Sorry, no guessing system yet. ", title="The neural network is guessing...")
        current_image = np.zeros((28, 28))
        print_guessed_number = False

    pygame.display.flip()
