

from kivy.app import App
# from kivy.uix.gridlayout import GridLayout
# from kivy.uix.label import Label
# from kivy.uix.textinput import TextInput
# from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder

Builder.load_file('testing.kv')
class Test(Widget): #GridLayout # was old
    name = ObjectProperty(None)
    favPizza = ObjectProperty(None)

    # #Intialize infinite keywords
    # def __init__(self, **kwargs):
    #     #Call grid layout constructor
    #     super(Test, self).__init__(**kwargs)
    #     #Set columns
    #     self.cols = 1
    #     #Set height and with of the grid
    #     self.row_force_default=True # forces the rows to be a default height
    #     self.row_default_height=120 # sets the default height of the rows
    #     self.col_force_default=True # forces the columns to be a default width
    #     self.col_default_width=100 # sets the default width of the columns

    #     # Create a second grid layout
    #     self.topGrid = GridLayout(row_force_default=True, # forces the rows to be a default height
    #                               row_default_height=40, # sets the default height of the rows
    #                               col_force_default=True, # forces the columns to be a default width
    #                               col_default_width=400) # sets the default width of the columns
    #     self.topGrid.cols = 2
    #     #add widgets
    #     self.topGrid.add_widget(Label(text='Name: '))
    #     #add text input-box
    #     self.name = TextInput(multiline=False)
    #     self.topGrid.add_widget(self.name)


    #     self.topGrid.add_widget(Label(text='favorite pizza: '))

    #     self.favPizza = TextInput(multiline=False)
    #     self.topGrid.add_widget(self.favPizza)

    #     # kannst unendlich lang wiederholen.

    #     #add the second grid to the first (so that you can see it)
    #     self.add_widget(self.topGrid)

    #     # add submit button
    #     self.submit = Button(text='Submit', font_size=32, size_hint_y=None, height=50, size_hint_x=None, width=200) # size_hint_y=None, height=50 makes the button a fixed size of y # size_hint_x=None, width=200 makes the button a fixed size of x
    #     #Bind the button
    #     self.submit.bind(on_press=self.press)
    #     self.add_widget(self.submit)
    
    # Define the press function
    def press(self):
        #grab the input
        name = self.name.text
        pizza = self.favPizza.text
        #print to console:
        print(f'Hello {name}. You like {pizza} pizza.')
        #Print to app
        #self.add_widget(Label(text=f'Hello {name}. You like {pizza} pizza.'))
        
        #Clear the input boxes
        self.name.text = ''
        self.favPizza.text = ''


class MyApp(App):
    def build(self):
        return Test()


if __name__ == '__main__':
    MyApp().run()