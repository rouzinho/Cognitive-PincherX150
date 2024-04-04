"""
Module storing the implementation of a circular progress bar in kivy.

.. note::

    Refer to the in-code documentation of the class and its methods to learn about the tool. Includes a usage example.

Authorship: Kacper Florianski
"""

from curses.textpad import rectangle
from kivy.uix.widget import Widget
from kivy.app import App
from kivy.core.text import Label
from kivy.lang.builder import Builder
from kivy.graphics import Line, Rectangle, Color, Ellipse
from kivy.clock import Clock
from collections.abc import Iterable
from math import ceil

# This constant enforces the cap argument to be one of the caps accepted by the kivy.graphics.Line class
_ACCEPTED_BAR_CAPS = {"round", "none", "square"}

# Declare the defaults for the modifiable values
_DEFAULT_THICKNESS = 10
_DEFAULT_CAP_STYLE = 'round'
_DEFAULT_PRECISION = 10
_DEFAULT_PROGRESS_ERROR_COLOUR = (0.9, 0.0, 0.368, 0.9)
_DEFAULT_PROGRESS_LP_COLOUR = (0.0, 0.90, 0.0, 0.9)
_DEFAULT_BACKGROUND_COLOUR = (0.26, 0.26, 0.26, 0.3)
_DEFAULT_MAX_PROGRESS = 100
_DEFAULT_MIN_PROGRESS = 0
_DEFAULT_WIDGET_SIZE = 200
_DEFAULT_TEXT_LABEL = Label(text="{}%", font_size=40, color=(0,0,0,0.7))

# Declare the defaults for the normalisation function, these are used in the textual representation (multiplied by 100)
_NORMALISED_MAX = 1
_NORMALISED_MIN = 0


class CircularProgressBar(Widget):
    """
    Widget used to create a circular progress bar.

    You can either modify the values within the code directly, or use the .kv language to pass them to the class.

    The following keyword values are currently used:

        1. thickness - thickness of the progress bar line (positive integer)
        2. cap_style - cap / edge of the bar, check the cap keyword argument in kivy.graphics.Line
        3. cap_precision - bar car sharpness, check the cap_precision keyword argument in kivy.graphics.Line
        4. progress_colour - Colour value of the progress bar, check values accepted by kivy.graphics.Color
        5. background_colour - Colour value of the background bar, check values accepted by kivy.graphics.Color
        6. max - maximum progress (value corresponding to 100%)
        7. min - minimum progress (value corresponding to 0%) - note that this sets the starting value to this value
        8. value - progress value, can you use it initialise the bar to some other progress different from the minimum
        9. widget_size - size of the widget, use this to avoid issues with size, width, height etc.
        10. label - kivy.graphics.Label textually representing the progress - pass a label with an empty text field to
        remove it, use "{}" as the progress value placeholder (it will be replaced via the format function)
        11. value_normalized - get the current progress but normalised, or set it using a normalised value

    .. note::

        You can execute this module to have a live example of the widget.

    .. warning::

        Apart from throwing kivy-specific errors, this class will throw TypeError and ValueError exceptions.

    Additionally, this class provides aliases to match the kivy.uix.progressbar.ProgressBar naming convention:

        1. get_norm_value - alternative name for get_normalised_progress
        2. set_norm_value - alternative name for set_normalised_progress
    """

    def __init__(self, **kwargs):
        super(CircularProgressBar, self).__init__(**kwargs)

        # Initialise the values modifiable via the class properties
        self._thickness = _DEFAULT_THICKNESS
        self._cap_style = _DEFAULT_CAP_STYLE
        self._cap_precision = _DEFAULT_PRECISION
        self._progress_colour = _DEFAULT_PROGRESS_LP_COLOUR
        self._progress_colour_error = _DEFAULT_PROGRESS_ERROR_COLOUR
        self._background_colour = _DEFAULT_BACKGROUND_COLOUR
        self._background_active = _DEFAULT_BACKGROUND_COLOUR
        self._background_non_active = _DEFAULT_BACKGROUND_COLOUR
        self._max_progress = _DEFAULT_MAX_PROGRESS
        self._min_progress = _DEFAULT_MIN_PROGRESS
        self._widget_size = _DEFAULT_WIDGET_SIZE
        self._text_label = _DEFAULT_TEXT_LABEL
        self._text_error = _DEFAULT_TEXT_LABEL
        self._text_error_string = _DEFAULT_TEXT_LABEL
        self._text_lp = _DEFAULT_TEXT_LABEL
        self._text_goal = _DEFAULT_TEXT_LABEL
        self._text_object = _DEFAULT_TEXT_LABEL

        # Initialise the progress value to the minimum - gets overridden post init anyway
        self._value = _DEFAULT_MIN_PROGRESS
        self._value_error = _DEFAULT_MIN_PROGRESS
        self._value_error_string = _DEFAULT_MIN_PROGRESS
        self._value_lp = _DEFAULT_MIN_PROGRESS
        self._value_goal = _DEFAULT_MIN_PROGRESS
        self._value_object = _DEFAULT_MIN_PROGRESS

        # Store some label-related values to access them later
        self._default_label_text = _DEFAULT_TEXT_LABEL.text
        self._default_label_error = _DEFAULT_TEXT_LABEL.text
        self._default_label_error_string = _DEFAULT_TEXT_LABEL.text
        self._default_label_lp = _DEFAULT_TEXT_LABEL.text
        self._default_label_goal = _DEFAULT_TEXT_LABEL.text
        self._default_label_object = _DEFAULT_TEXT_LABEL.text
        self._label_size = (0, 0)
        self._error_size = (0, 0)
        self._error_size_string = (0, 0)
        self._goal_size = (0, 0)
        self._lp_size = (0, 0)

        # Create some aliases to match the progress bar method names
        self.get_norm_value = self.get_normalised_progress
        self.set_norm_value = self.set_normalised_progress
        self.get_norm_value_error = self.get_normalised_progress_error
        self.set_norm_value_error = self.set_normalised_progress_error
        self.get_norm_goal = self.get_normalised_goal
        self.set_norm_goal = self.set_normalised_goal
        self.get_norm_error_string = self.get_normalised_error_string
        self.set_norm_error_string = self.set_normalised_error_string
        self.get_norm_lp = self.get_normalised_lp
        self.set_norm_lp = self.set_normalised_lp

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        if type(value) != int:
            raise TypeError("Circular bar thickness only accepts an integer value, not {}!".format(type(value)))
        elif value <= 0:
            raise ValueError("Circular bar thickness must be a positive integer, not {}!".format(value))
        else:
            self._thickness = value

    @property
    def cap_style(self):
        return self._cap_style

    @cap_style.setter
    def cap_style(self, value: str):
        if type(value) != str:
            raise TypeError("Bar line cap argument must be a string, not {}!".format(type(value)))
        value = value.lower().strip()
        if value not in _ACCEPTED_BAR_CAPS:
            raise ValueError("Bar line cap must be included in {}, and {} is not!".format(_ACCEPTED_BAR_CAPS, value))
        else:
            self._cap_style = value

    @property
    def cap_precision(self):
        return self._cap_precision

    @cap_precision.setter
    def cap_precision(self, value: int):
        if type(value) != int:
            raise TypeError("Circular bar cap precision only accepts an integer value, not {}!".format(type(value)))
        elif value <= 0:
            raise ValueError("Circular bar cap precision must be a positive integer, not {}!".format(value))
        else:
            self._cap_precision = value

    @property
    def progress_colour(self):
        return self._progress_colour

    @progress_colour.setter
    def progress_colour(self, value: Iterable):
        if not isinstance(value, Iterable):
            raise TypeError("Bar progress colour must be iterable (e.g. list, tuple), not {}!".format(type(value)))
        else:
            self._progress_colour = value

    @property
    def progress_colour_error(self):
        return self._progress_colour_error

    @progress_colour_error.setter
    def progress_colour_error(self, value: Iterable):
        if not isinstance(value, Iterable):
            raise TypeError("Bar progress colour must be iterable (e.g. list, tuple), not {}!".format(type(value)))
        else:
            self._progress_colour_error = value

    def set_color_bars(self,error,lp):
        pass

    @property
    def background_colour(self):
        return self._background_colour

    @background_colour.setter
    def background_colour(self, value: Iterable):
        if not isinstance(value, Iterable):
            raise TypeError("Bar background colour must be iterable (e.g. list, tuple), not {}!".format(type(value)))
        else:
            self._background_colour = value

    @property
    def background_active(self):
        return self._background_active

    @background_active.setter
    def background_active(self, value: Iterable):
        if not isinstance(value, Iterable):
            raise TypeError("Bar background colour must be iterable (e.g. list, tuple), not {}!".format(type(value)))
        else:
            self._background_active = value

    @property
    def background_non_active(self):
        return self._background_non_active

    @background_non_active.setter
    def background_non_active(self, value: Iterable):
        if not isinstance(value, Iterable):
            raise TypeError("Bar background colour must be iterable (e.g. list, tuple), not {}!".format(type(value)))
        else:
            self._background_non_active = value

    @property
    def max(self):
        return self._max_progress

    @max.setter
    def max(self, value: int):
        if type(value) != int:
            raise TypeError("Maximum progress only accepts an integer value, not {}!".format(type(value)))
        elif value <= self._min_progress:
            raise ValueError("Maximum progress - {} - must be greater than minimum progress ({})!"
                             .format(value, self._min_progress))
        else:
            self._max_progress = value

    @property
    def min(self):
        return self._min_progress

    @min.setter
    def min(self, value: int):
        if type(value) != int:
            raise TypeError("Minimum progress only accepts an integer value, not {}!".format(type(value)))
        elif value > self._max_progress:
            raise ValueError("Minimum progress - {} - must be smaller than maximum progress ({})!"
                             .format(value, self._max_progress))
        else:
            self._min_progress = value
            self._value = value
            self._value_error = value
            self._value_goal = value

    @property
    def value(self):
        return self._value

    @property
    def value_error(self):
        return self._value_error

    @property
    def value_goal(self):
        return self._value_goal

    @property
    def value_object(self):
        return self._value_object

    @property
    def value_error_string(self):
        return self._value_error_string

    @property
    def value_lp(self):
        return self._value_lp

    @value.setter
    def value(self, value: int):
        if type(value) != int:
            raise TypeError("Progress must be an integer value, not {}!".format(type(value)))
        elif self._min_progress > value or value > self._max_progress:
            raise ValueError("Progress must be between minimum ({}) and maximum ({}), not {}!"
                             .format(self._min_progress, self._max_progress, value))
        elif value != self._value:
            self._value = value
            self._draw()

    @value_error.setter
    def value_error(self, value: int):
        if type(value) != int:
            raise TypeError("Progress must be an integer value, not {}!".format(type(value)))
        elif self._min_progress > value or value > self._max_progress:
            raise ValueError("Progress must be between minimum ({}) and maximum ({}), not {}!"
                             .format(self._min_progress, self._max_progress, value))
        elif value != self._value:
            self._value_error = value
            self._draw()
    
    @value_error_string.setter
    def value_error_string(self, value: str):
        #if type(value) != int:
        #    raise TypeError("Progress must be an integer value, not {}!".format(type(value)))
        #elif self._min_progress > value or value > self._max_progress:
        #    raise ValueError("Progress must be between minimum ({}) and maximum ({}), not {}!"
        #                     .format(self._min_progress, self._max_progress, value))
        #elif value != self._value:
        self._value_error_string = value
        self._draw()

    @value_goal.setter
    def value_goal(self, value: str):
        #if type(value) != int:
        #    raise TypeError("Progress must be an integer value, not {}!".format(type(value)))
        #elif self._min_progress > value or value > self._max_progress:
        #    raise ValueError("Progress must be between minimum ({}) and maximum ({}), not {}!"
        #                     .format(self._min_progress, self._max_progress, value))
        #elif value != self._value:
        self._value_goal = value
        self._draw()

    @value_lp.setter
    def value_lp(self, value: str):
        #if type(value) != int:
        #    raise TypeError("Progress must be an integer value, not {}!".format(type(value)))
        #elif self._min_progress > value or value > self._max_progress:
        #    raise ValueError("Progress must be between minimum ({}) and maximum ({}), not {}!"
        #                     .format(self._min_progress, self._max_progress, value))
        #elif value != self._value:
        self._value_lp = value
        self._draw()

    @value_object.setter
    def value_object(self, value: int):
        #if type(value) != int:
        #    raise TypeError("Progress must be an integer value, not {}!".format(type(value)))
        #elif self._min_progress > value or value > self._max_progress:
        #    raise ValueError("Progress must be between minimum ({}) and maximum ({}), not {}!"
        #                     .format(self._min_progress, self._max_progress, value))
        #elif value != self._value:
        self._value_object = value

    @property
    def widget_size(self):
        return self._widget_size

    @widget_size.setter
    def widget_size(self, value: int):
        if type(value) != int:
            raise TypeError("Size of this widget must be an integer value, not {}!".format(type(value)))
        elif value <= 0:
            raise ValueError("Size of this widget must be a positive integer, not {}!".format(value))
        else:
            self._widget_size = value

    @property
    def label(self):
        return self._text_label

    @property
    def label_error(self):
        return self._text_error

    @property
    def label_error_string(self):
        return self._text_error_string

    @property
    def label_goal(self):
        return self._text_goal

    @property
    def label_lp(self):
        return self._text_lp

    @label.setter
    def label(self, value: Label):
        if not isinstance(value, Label):
            raise TypeError("Label must a kivy.graphics.Label, not {}!".format(type(value)))
        else:
            self._text_label = value
            self._default_label_text = value.text

    @label.setter
    def label_error(self, value: Label):
        if not isinstance(value, Label):
            raise TypeError("Label must a kivy.graphics.Label, not {}!".format(type(value)))
        else:
            self._text_error = value
            self._default_label_error = value.text

    @label.setter
    def label_error_string(self, value: Label):
        if not isinstance(value, Label):
            raise TypeError("Label must a kivy.graphics.Label, not {}!".format(type(value)))
        else:
            self._text_error_string = value
            self._default_label_error_string = value.text

    @label.setter
    def label_goal(self, value: Label):
        if not isinstance(value, Label):
            raise TypeError("Label must a kivy.graphics.Label, not {}!".format(type(value)))
        else:
            self._text_goal = value
            self._default_label_goal = value.text

    @label.setter
    def label_lp(self, value: Label):
        if not isinstance(value, Label):
            raise TypeError("Label must a kivy.graphics.Label, not {}!".format(type(value)))
        else:
            self._text_lp = value
            self._default_label_lp = value.text

    @property
    def value_normalized(self):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        return self.get_normalised_progress()

    @value_normalized.setter
    def value_normalized(self, value):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        self.set_normalised_progress(value)

    @property
    def value_normalized_error(self):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        return self.get_normalised_progress_error()

    @value_normalized.setter
    def value_normalized_error(self, value):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        self.set_normalised_progress_error(value)

    @property
    def value_normalized_error_string(self):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        return self.get_normalised_progress_error_string()

    @value_normalized_error_string.setter
    def value_normalized_error_string(self, value):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        self.set_normalised_progress_error_string(value)

    @property
    def value_normalized_lp(self):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        return self.get_normalised_lp()

    @value_normalized_lp.setter
    def value_normalized_lp(self, value):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        self.set_normalised_lp(value)

    @property
    def value_normalized_goal(self):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        return self.get_normalised_goal()

    @value_normalized_goal.setter
    def value_normalized_goal(self, value):
        """
        Alias the for getting the normalised progress.

        Matches the property name in kivy.uix.progressbar.ProgressBar.

        :return: Current progress normalised to match the percentage constants
        """
        self.set_normalised_goal(value)

    def _refresh_text(self):
        """
        Function used to refresh the text of the progress label.

        Additionally updates the variable tracking the label's texture size
        """
        #self._text_label.text = self._default_label_text.format(str(int(self.get_normalised_progress_error() * 100)))
        #self._text_error.text = self._default_label_error.format(str(int(self.get_normalised_progress() * 100)))
        self._text_error_string.text = self._default_label_error_string.format(self.get_normalised_error_string())
        self._text_lp.text = self._default_label_lp.format(self.get_normalised_lp())
        self._text_goal.text = self._default_label_goal.format(self.get_normalised_goal())
        #self._text_label.refresh()
        #self._text_error.refresh()
        self._text_goal.refresh()
        self._text_error_string.refresh()
        self._text_lp.refresh()
        #self._label_size = self._text_label.texture.size
        #self._error_size = self._text_error.texture.size
        self._goal_size = self._text_goal.texture.size
        self._error_size_string = self._text_error_string.texture.size
        self._lp_size = self._text_lp.texture.size

    def get_normalised_progress(self) -> float:
        """
        Function used to normalise the progress using the MIN/MAX normalisation

        :return: Current progress normalised to match the percentage constants
        """
        return _NORMALISED_MIN + (self._value - self._min_progress) * (_NORMALISED_MAX - _NORMALISED_MIN) \
            / (self._max_progress - self._min_progress)

    def get_normalised_progress_error(self) -> float:
        """
        Function used to normalise the progress using the MIN/MAX normalisation

        :return: Current progress normalised to match the percentage constants
        """
        return _NORMALISED_MIN + (self._value_error - self._min_progress) * (_NORMALISED_MAX - _NORMALISED_MIN) \
            / (self._max_progress - self._min_progress)

    def get_normalised_error_string(self) -> str:
        """
        Function used to normalise the progress using the MIN/MAX normalisation

        :return: Current progress normalised to match the percentage constants
        """
        return self._value_error_string  

    def get_normalised_lp(self) -> str:
        """
        Function used to normalise the progress using the MIN/MAX normalisation

        :return: Current progress normalised to match the percentage constants
        """
        return self._value_lp    

    def get_normalised_goal(self) -> str:
        """
        Function used to normalise the progress using the MIN/MAX normalisation

        :return: Current progress normalised to match the percentage constants
        """
        #return _NORMALISED_MIN + (self._value_goal - self._min_progress) * (_NORMALISED_MAX - _NORMALISED_MIN) \
        #    / (self._max_progress - self._min_progress)
        return self._value_goal

    def set_normalised_progress(self, norm_progress: int):
        """
        Function used to set the progress value from a normalised value, using MIN/MAX normalisation

        :param norm_progress: Normalised value to update the progress with
        """
        if type(norm_progress) != float and type(norm_progress) != int:
            raise TypeError("Normalised progress must be a float or an integer, not {}!".format(type(norm_progress)))
        elif _NORMALISED_MIN > norm_progress or norm_progress > _NORMALISED_MAX:
            raise ValueError("Normalised progress must be between the corresponding min ({}) and max ({}), {} is not!"
                             .format(_NORMALISED_MIN, _NORMALISED_MAX, norm_progress))
        else:
            self.value = ceil(self._min_progress + (norm_progress - _NORMALISED_MIN) *
                              (self._max_progress - self._min_progress) / (_NORMALISED_MAX - _NORMALISED_MIN))

    def set_normalised_progress_error(self, norm_progress: int):
        """
        Function used to set the progress value from a normalised value, using MIN/MAX normalisation

        :param norm_progress: Normalised value to update the progress with
        """
        if type(norm_progress) != float and type(norm_progress) != int:
            raise TypeError("Normalised progress must be a float or an integer, not {}!".format(type(norm_progress)))
        elif _NORMALISED_MIN > norm_progress or norm_progress > _NORMALISED_MAX:
            raise ValueError("Normalised progress must be between the corresponding min ({}) and max ({}), {} is not!"
                             .format(_NORMALISED_MIN, _NORMALISED_MAX, norm_progress))
        else:
            self.value_error = ceil(self._min_progress + (norm_progress - _NORMALISED_MIN) *
                              (self._max_progress - self._min_progress) / (_NORMALISED_MAX - _NORMALISED_MIN))

    def set_normalised_goal(self, norm_progress: str):
        """
        Function used to set the progress value from a normalised value, using MIN/MAX normalisation

        :param norm_progress: Normalised value to update the progress with
        """
        #if type(norm_progress) != float and type(norm_progress) != int:
        #    raise TypeError("Normalised progress must be a float or an integer, not {}!".format(type(norm_progress)))
        #elif _NORMALISED_MIN > norm_progress or norm_progress > _NORMALISED_MAX:
        #    raise ValueError("Normalised progress must be between the corresponding min ({}) and max ({}), {} is not!"
        #                     .format(_NORMALISED_MIN, _NORMALISED_MAX, norm_progress))
        #else:
        self.value_goal = norm_progress#ceil(self._min_progress + (norm_progress - _NORMALISED_MIN) *
                           #   (self._max_progress - self._min_progress) / (_NORMALISED_MAX - _NORMALISED_MIN))

    def set_normalised_error_string(self, norm_progress: str):
        """
        Function used to set the progress value from a normalised value, using MIN/MAX normalisation

        :param norm_progress: Normalised value to update the progress with
        """
        self.value_error_string = norm_progress

    def set_normalised_lp(self, norm_progress: str):
        """
        Function used to set the progress value from a normalised value, using MIN/MAX normalisation

        :param norm_progress: Normalised value to update the progress with
        """
        self.value_lp = norm_progress

    def _draw(self):
        """
        Function used to draw the progress bar onto the screen.

        The drawing process is as follows:

            1. Clear the canvas
            2. Draw the background progress line (360 degrees)
            3. Draw the actual progress line (N degrees where n is between 0 and 360)
            4. Draw the textual representation of progress in the middle of the circle
        """

        with self.canvas:
            self.canvas.clear()
            self._refresh_text()

            #Draw active widget
            Color(*self.background_active)
            Line(rectangle=(self.pos[0]-5,self.pos[1]-5,self.widget_size+10,self.widget_size+35))

            # Draw the background progress line
            Color(*self.background_colour)
            Line(circle=(self.pos[0] + self._widget_size / 2, self.pos[1] + self._widget_size / 1,
                         self._widget_size / 2 - self._thickness), width=self._thickness)

            # Draw the progress line
            Color(*self.progress_colour_error)
            Line(circle=(self.pos[0] + self._widget_size / 2, self.pos[1] + self._widget_size / 1,
                         self._widget_size / 2 - self._thickness, 0, self.get_normalised_progress_error() * 360),
                 width=self._thickness, cap=self._cap_style, cap_precision=self._cap_precision)

            # Center and draw the progress text
            Color(0.8,0.0,0.0,1)
            Rectangle(texture=self._text_error_string.texture, size=self._error_size_string,
                      pos=(self._widget_size / 2 - self._error_size_string[0] / 2 + self.pos[0],
                           self._widget_size / 0.6 - self._error_size_string[1] / 2 + self.pos[1]))

            # Draw the background progress line
            Color(*self.background_colour)
            Line(circle=(self.pos[0] + self._widget_size / 2, self.pos[1] + self._widget_size / 1,
                         self._widget_size / 3 - self._thickness), width=self._thickness)

            # Draw the progress line
            Color(*self.progress_colour)
            Line(circle=(self.pos[0] + self._widget_size / 2, self.pos[1] + self._widget_size / 1,
                         self._widget_size / 3 - self._thickness, 0, self.get_normalised_progress() * 360),
                 width=self._thickness, cap=self._cap_style, cap_precision=self._cap_precision)

            # Center and draw the progress text
            Color(0.0,0.4,0.2,1)
            Rectangle(texture=self._text_lp.texture, size=self._lp_size,
                      pos=(self._widget_size / 2 - self._lp_size[0] / 2 + self.pos[0],
                           self._widget_size / 1 - self._lp_size[1] / 2 + self.pos[1]))


            Color(0.933,0.902,0.807,1)
            Rectangle(texture=self._text_goal.texture, size=self._goal_size,
                      pos=(self._widget_size / 2 - self._goal_size[0] / 2 + self.pos[0],
                           self._widget_size / 3 - self._goal_size[1] / 3 + self.pos[1]))

            if self.value_object == 0:
                Color(0,0.4,0,0.9)
                Ellipse(pos=(self._widget_size  - self._goal_size[0] + self.pos[0] -15,
                           self._widget_size / 10 - self._goal_size[1] / 10 + self.pos[1]), 
                           size=(25, 25))
            if self.value_object == 1:
                Color(0.0,0.4,0.0,0.9)
                Ellipse(pos=(self._widget_size  - self._goal_size[0] + self.pos[0] -15,
                           self._widget_size / 10 - self._goal_size[1] / 10 + self.pos[1]), 
                           size=(25, 25))
            if self.value_object == 2:
                Color(0.0,0.4,0.0,0.9)
                Ellipse(pos=(self._widget_size  - self._goal_size[0] + self.pos[0] -15,
                           self._widget_size / 10 - self._goal_size[1] / 10 + self.pos[1]), 
                           size=(25, 25))

            Color(*self.background_non_active)
            Rectangle(size=(self._widget_size,self._widget_size+100),
                      pos=(self.pos[0],self.pos[1] + 10))


class _Example(App):

    # Simple animation to show the circular progress bar in action
    def animate(self, dt):
        for bar in self.root.children[:-1]:
            if bar.value < bar.max:
                bar.value += 1
            else:
                bar.value = bar.min

        # Showcase that setting the values using value_normalized property also works
        #bar = self.root.children[-1]
        #if bar.value < bar.max:
        #    bar.value_normalized += 0.01
        #else:
        #    bar.value_normalized = 0

    # Simple layout for easy example
    def build(self):
        container = Builder.load_string('''
#:import Label kivy.core.text.Label           
#:set _label Label(text="\\nI am a label\\ninjected in kivy\\nmarkup string :)\\nEnjoy! --={}=--")
#:set _another_label Label(text="Loading...\\n{}%", font_size=16, color=(1,1,0.5,1), halign="center")
GridLayout:
    cols: 3
    CircularProgressBar:
        pos: 50, 100
        thickness: 5
        cap_style: "RouND"
        progress_colour: "010"
        background_colour: "001"
        cap_precision: 3
        max: 160
        min: 100
        widget_size: 300
        label: _label
    CircularProgressBar:
        pos: 400, 100
        thickness: 5
    CircularProgressBar:
        pos: 650, 100
        cap_style: "SqUArE"
        thickness: 5
        progress_colour: 0.8, 0.8, 0.5, 1
        cap_precision:100
        max: 100
        widget_size: 100
        label: _another_label''')

        # Animate the progress bar
        Clock.schedule_interval(self.animate, 0.05)
        return container


if __name__ == '__main__':
    _Example().run()
