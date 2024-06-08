from Helpers.Globals import RELEASE_MODE
from Helpers.ImageUtil import draw_balloon_bounding_boxes
from App.Setup import setup_app


def update(app):
    ''' Update the game logic of the app every frame. '''
    app.engine.useShader(1)

    for balloon in app.balloons:
        balloon.update()
        balloon.update_bbox_and_centroid(app.player)

    frame = app.capture.frame
    if not RELEASE_MODE and frame is not None:
        frame = draw_balloon_bounding_boxes(app.balloons, frame)

    for balloon in app.balloons:
        balloon.check_collision(app.capture.pose_dict)

    for i in range(len(app.balloons) - 1):
        balloon.check_balloon_collision(app.balloons[i:])

    for balloon in app.balloons:
        if balloon.despawn:
            app.sound_player.play_sound('balloon-pop.wav')
            balloon.respawn()
            balloon.update_bbox_and_centroid(app.player)

    if frame is not None:
        app.overlay.setTexture(frame)


if __name__ == '__main__':
    setup_app('ObjFiles/Balloon/balloon.obj', update)
