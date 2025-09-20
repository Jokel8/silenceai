from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button


class LoginScreen(GridLayout):
    #Intialize infinite keywords
    def __init__(self, **kwargs):
        #Call grid layout constructor
        super(LoginScreen, self).__init__(**kwargs)
        #Set columns
        self.cols = 2
        #add widgets
        self.add_widget(Label(text='Name: '))
        #add text input-box
        self.name = TextInput(multiline=False)
        self.add_widget(self.name)


        self.add_widget(Label(text='favorite pizza: '))

        self.favPizza = TextInput(multiline=False)
        self.add_widget(self.favPizza)

        # kannst unendlich lang wiederholen.

        # add submit button
        self.submit = Button(text='Submit', font_size=32)
        #Bind the button
        self.submit.bind(on_press=self.press)
        self.add_widget(self.submit)
    
    # Define the press function
    def press(self, instance):
        #grab the input
        name = self.name.text
        pizza = self.favPizza.text
        #print to console:
        #print(f'Hello {name}. You like {pizza} pizza.')
        #Print to app
        self.add_widget(Label(text=f'Hello {name}. You like {pizza} pizza.'))
        
        #Clear the input boxes
        self.name.text = ''
        self.favPizza.text = ''


class MyApp(App):

    def build(self):
        return LoginScreen()


if __name__ == '__main__':
    MyApp().run()