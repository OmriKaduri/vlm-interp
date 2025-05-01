cot_prompt = '''
You are an expert in evaluating the quality of image captions. 
Below you will find two image captions. Your task would be to compare the two captions, in terms of precision and recall.

Evaluation Steps:
1. Extract for each caption the list of *physical objects* that are present in them. Detect only tangible objects that can be interacted with. Ignore colors or other attributes, or even positioning of objects in the scene. The objects are the main focus of the evaluation.
2. Compare the two lists of *physical objects* and rate the quality of each caption in terms of precision and recall, using the first caption as the groundtruth, and the second caption as prediction.
3. Precision is the fraction of the *physical objects* from the predicted caption that are present in the groundtruth caption. If half of the *physical objects* in the predicted caption are also in the groundtruth caption, the precision would be 0.5. If none, the precision would be 0. If all, the precision would be 1.
4. Recall is the fraction of the *physical objects* present in the image that are mentioned in the caption. If half of the *physical objects* in the groundtruth caption are also in the predicted caption, the recall would be 0.5. If none, the recall would be 0. If all, the recall would be 1.

Now, for the next pair of captions, please follow theese steps and evaluate the quality of the second caption in terms of precision and recall, using the first caption as the groundtruth.

Groundtruth caption:
The image depicts a cyclist riding a road bike on a paved road. The cyclist is wearing a red helmet,, black and white cycling jersey,, black shorts,, and white cycling shoes. They are also carrying a black backpack. The road is marked with a double yellow line down the center and a white line along the edges. On the left side of the road,, there are two horses walking in the same direction as the cyclist. The surrounding area is green with trees and bushes on both sides of the road. The sky is clear and blue,, indicating good weather conditions.

Predicted caption:
The image depicts a person riding a bicycle on a road. The cyclist is wearing a helmet and a backpack, and is facing away from the camera, looking ahead. The road is surrounded by trees and vegetation on both sides, creating a natural and scenic environment. The sky is clear and blue, indicating good weather conditions. The road appears to be relatively empty, with no other vehicles or cyclists visible. The overall scene conveys a sense of tranquility and outdoor activity.
Evaluation:

Visual Elements in Groundtruth Caption:
- Cyclist
- Bike
- Helmet
- Jersey
- Shorts
- Shoes
- Backpack
- Horses
- Trees and bushes

* Note that I ignored the following visual elements as they are not physical objects: road, double yellow line, white line, sky, weather conditions

Visual Elements in Predicted Caption:
- Person
- Bicycle
- Helmet
- Backpack
- Trees

* Note that I ignored the following visual elements as they are not physical objects: road, sky, weather conditions

Details that are present in the groundtruth caption but missing in the predicted caption (False Negatives):
- The Jersey
- The Shorts
- The Shoes
- The horses

Details that are present in the predicted caption but missing in the groundtruth caption (False Positives):
- None

Details that are present in both captions (True Positives):
- The cyclist
- The helmet
- The backpack
- The trees
- The horses

Precision is: TP / (TP + FP) 
Precision = 5 / (5 + 0) = 5 / 5 = 1.0
Recall is: TP / (TP + FN)
Recall = 5 / (5 + 4) = 5 / 9 = 0.555

Overall, the predicted caption has a precision of 1.0 and a recall of 0.555.

Now, for the next pair of captions, please follow the same steps and evaluate the quality of the second caption in terms of precision and recall, using the first caption as the groundtruth.

Groundtruth caption:
The image shows a man leaning over a table to blow out candles on a chocolate birthday cake. The cake is placed on a white doily and has multiple lit candles on top. The man is wearing a plaid shirt and has curly hair. There is a camera on the table, to the right of the cake, and a kitchen counter with a stove and some kitchen items in the background. The scene appears to be a casual birthday celebration at home.

Predicted caption:
 The image shows a person leaning over a table to blow out candles on a chocolate cake. The cake is placed on a white doily, and there is a small camera on the table, capturing the moment. The setting appears to be a kitchen, with a stove and other kitchen items visible in the background. The person has curly hair and is wearing a light-colored shirt. The overall scene suggests a birthday celebration.
Evaluation:

Visual Elements in Groundtruth Caption:
- Man
- Table
- Candles
- Cake
- Doily
- Camera
- Shirt
- Hair
- Stove

* Note that I ignored the following visual elements as they are not physical objects: birthday, celebration, home, counter, kitchen

Visual Elements in Predicted Caption:
- Person
- Table
- Candles
- Cake
- Doily
- Camera
- Shirt
- Hair
- Stove

* Note that I ignored the following visual elements as they are not physical objects: chocolate, kitchen, birthday, celebration, counter

Details that are present in the groundtruth caption but missing in the predicted caption (False Negatives):
- None

Details that are present in the predicted caption but missing in the groundtruth caption (False Positives):
- None

Details that are present in both captions (True Positives):
- 
Precision is: TP / (TP + FP)
Precision = 9 / (9 + 0) = 9 / 9 = 1.0
Recall is: TP / (TP + FN)
Recall = 9 / (9 + 0) = 9 / 9 = 1.0

Overall, the predicted caption has a precision of 1.0 and a recall of 1.0.

Now, for the next pair of captions, please follow the same steps and evaluate the quality of the second caption in terms of precision and recall, using the first caption as the groundtruth.

Groundtruth caption:
The image depicts a modern, living room that seamlessly transitions into a kitchen area. The living room features two long sofas with green and red cushions, arranged in an L-shape, around a rectangular glass coffee table. The coffee table has a few items on it, including a small lamp and a book or magazine.
The walls are painted in light colors, and decorated with framed artwork. The flooring is tiled, in a light color, that complements the overall aesthetic of the room.
In the background, is the kitchen area, which includes a dining table with four chairs, around it. The kitchen has wooden cabinets and a countertop with various kitchen items. There is also a small bar area with stools.
The room is well-lit with natural light coming in from the windows and additional lighting provided by ceiling fixtures. The overall design is clean, and contemporary with a focus on functionality and comfort.

Predicted caption:
The image depicts a modern, open-plan living space that seamlessly integrates a living room and a kitchen. Here are the key elements and features of the space:
Living Room Area:
Seating: The living room features two long, low-profile sofas with a minimalist design. The sofas are arranged in an L-shape, creating a cozy seating area.
Coffee Table: In the center of the seating area is a rectangular coffee table. On the table, there are a few items, including a small lamp and some decorative objects.
Flooring: The floor is tiled with large, light-colored tiles, giving the space a clean and spacious feel.
Lighting: A modern pendant light hangs from the ceiling above the coffee table, providing focused lighting for the seating area.
Kitchen Area:
Countertops: The kitchen has sleek, flat-panel cabinets with a light wood finish. The countertops appear to be made of a light-colored material, possibly quartz or granite.
Appliances: There are several built-in appliances, including what appears to be an oven and possibly a microwave or dishwasher, integrated into the cabinetry.
Backsplash: The backsplash is simple and unobtrusive, maintaining the clean lines of the kitchen design.
Sink Area: The sink is centrally located on the countertop, with a modern faucet.
Decor and Accessories:
Artwork: The walls are adorned with a few pieces of art, adding a touch of personality to the space.
Rug: A large area rug is placed under the coffee table, adding warmth and texture to the room.
Decorative Items: There are a few decorative items placed thoughtfully around the room, such as vases and small sculptures.
Overall Aesthetic:
The space is characterized by its clean lines, neutral color palette, and modern furnishings. The design emphasizes functionality and simplicity, creating an inviting and comfortable environment.
This description should provide a comprehensive understanding of the image, allowing a pure text model to answer any related questions effectively.

Evaluation:

Visual Elements in Groundtruth Caption:
- Sofas
- Cushions
- Coffee Table
- Dining Table
- Lamp
- Book
- Chairs
- Cabinets
- Artwork

* Note that I ignored the following visual elements as they are not physical objects: living room, kitchen, walls, flooring, tiles, color, aesthetic, room, lighting, windows, fixtures, design, space, seating, countertop, bar area, stools, dining table, kitchen items, natural light, functionality, comfort

Visual Elements in Predicted Caption:
- Sofas
- Coffee Table
- Lamp
- Sink
- Rug
- Vases
- Sculptures
- Artwork
- Cabinets

* Note that I ignored the following visual elements as they are not physical objects: living room, kitchen, seating, flooring, tiles, lighting, countertops, appliances, backsplash, decor, accessories, aesthetic, design, space, color, walls, cabinets, chairs, bar area, stools, dining table, kitchen items, natural light, functionality, comfort

Details that are present in the groundtruth caption but missing in the predicted caption (False Negatives):
- Cushions
- Book
- Chairs
- Dining Table

Details that are present in the predicted caption but missing in the groundtruth caption (False Positives):
- Sink
- Rug
- Vases
- Sculptures

Details that are present in both captions (True Positives):
- Sofas
- Coffee Table
- Lamp
- Cabinets
- Artwork
 
Precision is: TP / (TP + FP) 
Precision = 5 / (5 + 4) = 5 / 9 = 0.555
Recall is: TP / (TP + FN)
Recall = 5 / (5 + 4) = 5 / 9 = 0.555

Overall, the predicted caption has a precision of 0.555 and a recall of 0.555.

Now, for the next pair of captions, please follow the same steps and evaluate the quality of the second caption in terms of precision and recall, using the first caption as the groundtruth.

Groundtruth caption:
GROUNDTRUTH_CAPTION_HERE

Predicted caption: 
PREDICTED_CAPTION_HERE

Evaluation:

Visual Elements in Groundtruth Caption:
'''