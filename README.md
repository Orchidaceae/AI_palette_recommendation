# AI_palette_recommendation
Recommendation system for 3 color palettes

Uses supervised learning in order to predict user rating of palettes, classifying palettes into 3 score classes: 1 2 3 where 1=dislike, 2=ok, 3=like.


# Work diary

## Try #1

### 3 dim palette space
We first tested setting up a model for a 3 dimentional palette space where the axis were each one integer color of the palette. The model learned poorly since the data became to separate which made it hard to measure distance between datapoints and made the classification impossible on a small data set. This could be due to the colors being a mix of the color channels red/green/blue and representing them as decimal integers would for example make two blueish colors with a bit separate red values very distant on a color axis since the most sigificant byte is the red byte of the 3 byte hex colors. This systematic separation could be learned by a network but would proboably require more data than we had access to, therefore we decided to try modelling our sample data in a different way.

## Try #2

### 9 dim palette space
