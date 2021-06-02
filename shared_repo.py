# -*- coding: utf-8 -*-

'''Import Libraries'''
import matplotlib.pyplot as plot
import logging

class SharedVisualizations():
    def __init__(self, x=None, y=None, y_test=None, y_pred=None):
        self.x = x
        self.y = y
        self.y_test = y_test
        self.y_pred = y_pred
    
    def twoD_plot(self, title=None, x_label=None, y_label=None):
        '''Comment:
           Function to create 2d plot visualization
        '''
        try:
            plot.scatter(self.x, self.y)
            plot.title(title)
            plot.xlabel(x_label)
            plot.ylabel(y_label)
            plot.show()
            return 'Plot visualized successfully'
        except Exception as e:
            logging.error(f'Error is: {e}')
            return e
        
    def actual_vs_predicted(self, title='Actual vs Predicted', x_label=None, y_label='Index'):
        '''Comment:
            Function to create actual vs predicted graph
        '''
        try:
            c = [i for i in range(1, len(self.y_test)+1, 1)] #generating index
            fig = plot.figure()
            plot.plot(c, self.y_test, color='blue', linewidth=2.5, linestyle='-')
            plot.plot(c, self.y_pred, color='red', linewidth=2.5, linestyle='-')
            fig.suptitle(title, fontsize=20)
            plot.xlabel(x_label, fontsize=18)
            plot.ylabel(y_label, fontsize=16)
            plot.show()
            return 'Plot visualized successfully'
        except Exception as e:
            logging.error(f'Error is: {e}')
            
    def error_term(self, title='Error Term', x_label=None, y_label='Index'):
        '''Comment:
            Function to create actual vs predicted graph
        '''
        try:
            c = [i for i in range(1, len(self.y_test)+1, 1)]
            fig = plot.figure()
            plot.plot(c,self.y_test-self.y_pred, color='blue', linewidth=2.5, linestyle='-')
            fig.suptitle(title, fontsize=20)
            plot.xlabel('Index', fontsize=18)
            plot.ylabel('y_test-y_pred', fontsize=16)
            plot.show()
            return 'Plot visualized successfully'
        except Exception as e:
            logging.error(f'Error is: {e}')
