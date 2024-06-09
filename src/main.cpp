
#include "Boids.h"
#include "Parameters.cuh"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define _USE_MATH_DEFINES
#include <math.h>

bool pause = false;
int windowWidth = 800;
int windowHeight = 800;
float boidSize = 4.0f;
Boids* boids = 0;
Shader* shader = 0;
GLFWwindow* window = 0;

int initGlfwAndGlad();
void initParams(Parameters& params);
void initShader();
void initImGui();
void renderImGui();
void resize(GLFWwindow* window, int width, int height);
void cleanup();

int main(int argc, char** argv)
{
	if (initGlfwAndGlad())
		return -1;
	int numBoids = argc > 1 ? atoi(argv[1]) : 10000;
	Parameters params{};
	params.algorithm = 0;
	if (argc > 2)
	{
		switch (atoi(argv[2]))
		{
		case 0:
			params.algorithm = 0;
			break;
		case 1:
			params.algorithm = 1;
			break;
		case 2:
			params.algorithm = 2;
			break;
		default:
			params.algorithm = 0;
		}
	}
	params.numBoids = numBoids;
	initParams(params);
	initShader();
	initImGui();
	glClearColor(0, 0, 0, 1);
	boids = new Boids(params);
	while (!glfwWindowShouldClose(window))
	{
		if (!pause)
			boids->compute();
		boids->render();
		renderImGui();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	cleanup();
	return 0;
}

int initGlfwAndGlad()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(windowWidth, windowHeight, "Boids", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, resize);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	return 0;
}

void initParams(Parameters& params)
{
	params.boxSize.x = windowWidth;
	params.boxSize.y = windowHeight;
	params.cohesionFactor = 0.0005f;
	params.allignmentFactor = 0.05f;
	params.separationFactor = 0.05f;
	params.escapeFactor = 0.1f;
	params.margin = 100.0f;
	params.marginFactor = 0.2f;
	params.viewRange = 40.0f;
	params.minDist = 8.0f;
	params.minSpeed = 3.0f;
	params.maxSpeed = 6.0f;
	params.gridSize = make_int2(windowWidth / (int)params.viewRange, windowHeight / (int)params.viewRange);
	params.cellSize.x = (float)params.boxSize.x / params.gridSize.x;
	params.cellSize.y = (float)params.boxSize.y / params.gridSize.y;
	params.numCells = params.gridSize.x * params.gridSize.y;
}

void initShader()
{
	shader = new Shader("vertexShader.txt", "fragmentShader.txt", "geometryShader.txt");
	shader->use();
	glm::mat4 p = glm::ortho(0.0f, (float)windowWidth, 0.0f, (float)windowHeight, -1.0f, 1.0f);
	shader->setMat4("projection", p);
	shader->setFloat("size", boidSize);
}

void initImGui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

void renderImGui()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGuiIO& io = ImGui::GetIO();
	if (io.MouseDown[1])
	{
		boids->params.mouseDown = true;
		auto pos = io.MousePos;
		boids->params.mousePosition = make_float2(pos.x, windowHeight - pos.y);
	}
	else
		boids->params.mouseDown = false;

	ImGui::Begin("Configuration");
	ImGui::Text("Application running in %.1f FPS", io.Framerate);
	ImGui::SliderFloat("Separation", &boids->params.separationFactor, 0.0f, 0.2f);
	ImGui::SliderFloat("Alignment", &boids->params.allignmentFactor, 0.0f, 0.2f);
	ImGui::SliderFloat("Cohesion", &boids->params.cohesionFactor, 0.0f, 0.02f);
	if (ImGui::SliderFloat("Boid size", &boidSize, 2.0f, 8.0f))
		shader->setFloat("size", boidSize);
	ImGui::Checkbox("Pause", &pause);
	if (ImGui::Button("Reset boids"))
		boids->reset();
	if (ImGui::Button("Reset params"))
		boids->resetParams();
	boids->copyParamsToDevice();
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void resize(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
	int2 boxSize = make_int2(width, height);
	boids->changeBoxSize(boxSize);
	windowWidth = width;
	windowHeight = height;
	glm::mat4 p = glm::ortho(0.0f, (float)width, 0.0f, (float)height, -1.0f, 1.0f);
	shader->setMat4("projection", p);
}

void cleanup()
{
	delete boids;
	delete shader;
	glfwTerminate();
}