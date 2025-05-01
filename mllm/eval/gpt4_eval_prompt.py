GPT4EvalPrompt = '''
You are an expert in evaluating the quality of image captions. Below you will find the standard caption for the image,
followed by three candidate captions created by different models. Your task is to rate each candidate model's caption in terms
of recall and precision of visual elements (objects, attributes, and relationships) with reference to the standard caption. Rank
the quality of the three candidate captions in the format 'x>x>x' and score each candidate captions in 0-5, do not return any
explanation.

Example 1.
Standard caption:
The image depicts an overhead view of a tennis player serving. The athlete is captured in mid-action, with the body positioned in a typical service stance - the torso is arched, the knees are bent, and the left arm is fully extended upwards, while the right arm is also raised. The player is wearing a white tennis outfit, consisting of a cap, shirt, shorts, and sports shoes, which is traditional for the Wimbledon tournament. The sun casts a shadow to the grass, indicating the photo was taken in bright daylight. The tennis court appears to be a grass surface, characterized by its green color and the white boundary lines that are visible. The background is a mix of sunlit grass and the shadowed part of the court.

Candidate caption 1:
The image features a man in a white shirt and white shorts playing tennis on a grass court. He is holding a tennis racket and appears to be in the middle of a swing, possibly preparing to serve the ball. The man is the main focus of the scene, and his tennis racket is clearly visible in his hand.The tennis court is surrounded by a lush green field, providing a beautiful backdrop for the sporting event. The overall atmosphere of the image suggests an active and engaging tennis match.

Candidate caption 2: 
In the heart of the image, a tennis player is captured in a moment of intense action. Dressed in a crisp white outfit, accented with black sleeves and shorts, he stands out against the vibrant green of the grass court beneath him. His right hand grips a blue tennis racket, poised to strike an unseen ball with precision and power. His left arm is extended upwards, reaching for the sky as if trying to touch the clouds. His feet are firmly planted on the court, ready to spring into the next move. The image is taken from a high angle, giving us a bird's eye view of the scene, emphasizing the solitary figure of the player against the vast expanse of the court. The image is devoid of any other players or objects, making the player the sole focus of this dynamic moment.

Candidate caption 3:
The image captures a moment during a tennis match, taken from an overhead perspective. A male tennis player is in the midst of serving the ball. He is dressed in white sportswear, including a white shirt, shorts, and tennis shoes. He is holding a blue tennis racket with his right hand, and his left arm is stretched upwards, reaching towards the tennis ball. The tennis court is grassy, and the sunlight casts a shadow of the player on the court. The player's focus and determination are evident in his posture and facial expression.

Evaluation Result:
Rank: 2>1>3
Candidate caption 1 score: 4.5
Candidate caption 2 score: 5.0
Candidate caption 3 score: 4.0

Example 2.

Standard caption:
The image depicts a cyclist riding a road bike on a paved road. The cyclist is wearing a red helmet,, black and white cycling jersey,, black shorts,, and white cycling shoes. They are also carrying a black backpack. The road is marked with a double yellow line down the center and a white line along the edges. On the left side of the road,, there are two horses walking in the same direction as the cyclist. The surrounding area is green with trees and bushes on both sides of the road. The sky is clear and blue,, indicating good weather conditions.

Candidate caption 1:
The image depicts a person riding a bicycle on a road. The rider is wearing a helmet, and has a backpack. The road is surrounded by greenery, on both sides, and there are two horses walking on the road ahead of the cyclist. The sky is clear and blue,,

Candidate caption 2:
In the image, there is a person wearing a shirt. The shirt is visible in the middle section of the image. The person appears to be standing or moving in a setting that includes trees and possibly a path or road. The shirt is the main focus in this description.

Candidate caption 3:
The image depicts a person riding a bicycle on a road. The rider is wearing a helmet and appears to be in motion, suggesting they are actively cycling. The road is surrounded by greenery, indicating a natural or suburban setting. The weather seems to be clear, as there are no visible signs of rain or overcast skies. The overall scene conveys a sense of outdoor activity and possibly a leisurely or commuting ride.

Evaluation Result:
Rank: 1>3>2
Candidate caption 1 score: 4.5
Candidate caption 2 score: 2.5
Candidate caption 3 score: 3.5

Example 3.
Standard caption: The image shows an aerial view of a large parking lot filled with numerous cars. The parking lot is adjacent to a large building complex, and beyond the parking lot, there is a residential area with many houses. The view is from the window of an airplane, as the wing of the airplane is visible in the upper left corner of the image. The sky is clear with a few clouds, and the overall scene appears to be in a suburban or urban area.

Candidate caption 1:
The image shows an aerial view of a large area, as seen from an airplane window. The airplane's wing is prominently visible in the upper left corner of the image. Below, the scene includes a large parking lot filled with numerous cars, and a few buildings Beyond the parking lot, and buildings, are residential areas with many houses and green spaces. The sky is clear with a few clouds, and the overall scene suggests a suburban or urban area.
 
Candidate caption 2:
The image shows an airplane wing from a passenger's perspective, as the plane is flying over a city. The wing is visible on the right side of the image, and the cityscape below includes buildings, and streets. The sky is clear, and the overall scene suggests a typical urban area viewed from above during a flight.

Candidate caption 3:
The image shows an aerial view of a city or urban area. The perspective is from a high altitude, likely taken from an airplane or drone. The view includes a mix of residential and commercial buildings, roads, and possibly some green spaces. The image captures a dense urban environment with a variety of structures and infrastructure.

Evaluation Result:
Rank: 1>2>3
Candidate caption 1 score: 5.0
Candidate caption 2 score: 4.0
Candidate caption 3 score: 3.5

Now evaluate the following three captions according to the standard caption.

Standard caption:
GT_CAPTION

Candidate caption 1:
CANDIDATE_CAPTION_1

Candidate caption 2:
CANDIDATE_CAPTION_2

Candidate caption 3:
CANDIDATE_CAPTION_3

Evaluation Result:
Rank:

'''