#include <SFML/Graphics.hpp>
#include <chrono>
#include <cstdlib>
#include <cmath>

#if defined(NDEBUG)
#pragma comment(lib, "sfml-graphics-s.lib")
#pragma comment(lib, "sfml-system-s.lib")
#pragma comment(lib, "sfml-window-s.lib")
#else
#pragma comment(lib, "sfml-graphics-s-d.lib")
#pragma comment(lib, "sfml-system-s-d.lib")
#pragma comment(lib, "sfml-window-s-d.lib")
#endif

#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "opengl32.lib")

//SFML REQUIRED TO LAUNCH THIS CODE

#define SCALE 2
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define FIELD_WIDTH int(WINDOW_WIDTH / SCALE)
#define FIELD_HEIGHT int(WINDOW_HEIGHT / SCALE)

static struct Config
{
	float velocityDiffusion;
	float pressure;
	float vorticity;
	float colorDiffusion;
	float densityDiffusion;
	float forceScale;
	float bloomIntense;
	int radius;
	bool bloomEnabled;
} config;

void setConfig(float vDiffusion = 0.8f, float pressure = 1.5f, float vorticity = 50.0f, float cDiffuion = 0.8f,
	float dDiffuion = 1.2f, float force = 1000.0f, float bloomIntense = 25000.0f, int radius = 100, bool bloom = true);
void computeField(uint8_t* result, float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed);
void cudaInit(size_t xSize, size_t ySize);
void cudaExit();

int main()
{
	cudaInit(FIELD_WIDTH, FIELD_HEIGHT);
	srand(time(NULL));
	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "");
	window.setMouseCursorVisible(false);

	auto start = std::chrono::system_clock::now();
	auto end = std::chrono::system_clock::now();

	sf::Texture texture;
	sf::Sprite sprite;
	std::vector<sf::Uint8> pixelBuffer(FIELD_WIDTH * FIELD_HEIGHT * 4);
	texture.create(FIELD_WIDTH, FIELD_HEIGHT);

	sf::Vector2i mpos1 = { -1, -1 }, mpos2 = { -1, -1 };

	bool isPressed = false;
	bool isPaused = false;
	while (window.isOpen())
	{
		end = std::chrono::system_clock::now();
		std::chrono::duration<float> diff = end - start;
		window.setTitle("Fluid simulator " + std::to_string(int(1.0f / diff.count())) + " fps");
		start = end;

		window.clear(sf::Color::White);
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();

			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
			{
				window.close();
			}

			if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Button::Left)
				{
					mpos1 = { event.mouseButton.x, event.mouseButton.y };
					mpos1.x /= SCALE;
					mpos1.y /= SCALE;
					isPressed = true;
				}
				else
				{
					isPaused = !isPaused;
				}
			}
			if (event.type == sf::Event::MouseButtonReleased)
			{
				isPressed = false;
			}
			if (event.type == sf::Event::MouseMoved)
			{
				std::swap(mpos1, mpos2);
				mpos2 = { event.mouseMove.x, event.mouseMove.y };
				mpos2.x /= SCALE;
				mpos2.y /= SCALE;
			}
		}
		float dt = 0.02f;
		if (!isPaused)
		{
			computeField(pixelBuffer.data(), dt, mpos1.x, mpos1.y, mpos2.x, mpos2.y, isPressed);
		}

		texture.update(pixelBuffer.data());
		sprite.setTexture(texture);
		sprite.setScale({ SCALE, SCALE });
		window.draw(sprite);
		window.display();
	}
	cudaExit();
	return 0;
}