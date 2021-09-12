import pygame
import pygame_gui as pygg
import numpy as np
from engine import Engine 

pygame.init()
engine = Engine()
engine.load_settings("settings.yaml")
engine.debugging = True
CANVASLENGTH = 800
pxPerU = int(round(CANVASLENGTH / engine.config["boxSize"]))
RAD = int(round(pxPerU/2))
MAGFACTOR = 2
screen = pygame.display.set_mode((1200,CANVASLENGTH))
pygame.display.set_caption("SketchMM")
manager = pygg.UIManager((1200, CANVASLENGTH), theme_path="theme.json")
clock = pygame.time.Clock()

start_button = pygg.elements.UIButton(relative_rect=pygame.Rect((840, 10), (170, 50)), 
                                        text='Start', 
                                        manager=manager)
step_button = pygg.elements.UIButton(relative_rect=pygame.Rect((1020, 10), (170, 50)),
                                        text="Step",
                                        manager=manager)
                                        
info_block = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 70), (350, 50)), 
                                        text="FPS: 0.0    Time: 0.0    Iterations: 0.0", 
                                        manager=manager)
reset = pygg.elements.UIButton(relative_rect=pygame.Rect((840, 130), (350, 70)),
                                        text="Reset",
                                        manager=manager)
energy = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 210), (350,70)), manager=manager, text=" Energy: 0.0")
avgt = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 290), (350,70)), manager=manager, text=" Average T: 0.0")
instantt = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 370), (350,70)), manager=manager, text=" Instant T: 0.0")
avgp = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 450), (350,70)), manager=manager, text=" Average P: 0.0")
instantp = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 530), (350, 70)), manager=manager, text=" Instant P: 0.0")
normvel = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 610), (350, 70)), manager=manager, text=" Normalized Velocity: 0.0")
normacc = pygg.elements.ui_label.UILabel(relative_rect=pygame.Rect((840, 690), (350, 70)), manager=manager, text=" Normalized Acceleration: 0.0")



panel = pygame.Surface((400, 800))

simulating = False
running = True
while running:
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False 

        if event.type == pygame.USEREVENT:
            if event.user_type == pygg.UI_BUTTON_PRESSED:
                if event.ui_element == start_button:
                    if (simulating):
                        start_button.set_text("Resume")
                    else:
                        start_button.set_text("Pause")
                    simulating = not simulating

                elif event.ui_element == step_button:
                    engine.step_forward()
                
                elif event.ui_element == reset:
                    if not simulating:
                        engine.reset()

        manager.process_events(event)

    screen.fill((255, 255, 255))
    panel.fill((169, 169, 169))
    factvel = np.linalg.norm(engine.vel)
    factacc = np.linalg.norm(engine.acc)
    norm_vel = (engine.vel / factvel)
    
    for i in range(engine.config["N"]):
        px = int(round(engine.pos[i][0] * pxPerU))
        py = int(round(CANVASLENGTH - engine.pos[i][1] * pxPerU))
        # reverse y-direction of the arrow 
        vx = int(round(px+norm_vel[i][0]*MAGFACTOR*pxPerU))
        vy = int(round(py+norm_vel[i][1]*MAGFACTOR*pxPerU*-1))
        pygame.draw.circle(screen, (0, 0, 0), (px, py), RAD)
        pygame.draw.line(screen, (255, 0, 0), (px, py), (vx, vy), 3)
    
    if simulating:
        for i in range(5):
            engine.step_forward()
    info_block.set_text(f"FPS: {clock.get_fps():.1f}   Time: {engine.time:.2f}  Iterations: {engine.iterations}")
    energy.set_text(f"Energy: {engine.energy:.3f}")
    avgt.set_text(f"Average T: {engine.avgT:.3f}")
    instantt.set_text(f"Instant T: {engine.instantT:.3f}")
    avgp.set_text(f"Average P: {engine.avgP:.3f}")
    instantp.set_text(f"Instant P: {engine.instantP:.3f}")
    normvel.set_text(f"Normalized Velocity: {factvel:.2f}")
    normacc.set_text(f"Normalized Acceleration: {factacc:.2f}")

    screen.blit(panel, (830, 0))
    manager.update(time_delta)
    manager.draw_ui(screen)
    pygame.display.flip()

pygame.quit()