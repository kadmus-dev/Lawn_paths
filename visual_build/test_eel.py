from other import get_masks, get_shapefile
import eel


@eel.expose
def test(name):
    print(name)
    get_masks(name)
    get_shapefile(name)
    print("finished")
    eel.my_javascript_function('Finished ', 'job!')
