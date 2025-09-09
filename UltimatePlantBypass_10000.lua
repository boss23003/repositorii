-- Full Injectable Neural Network for Roblox (Robust, Serializable, Logger Mode)
-- Author: (you can add your name)
-- Date: 09.09.2025
-- Usage: Put this file on GitHub and optionally load with game:HttpGet + loadstring.
-- Safety: External load wrapped in pcall. Works standalone if external load fails.

-- CONFIG
local GITHUB_RAW_URL = "https://raw.githubusercontent.com/YourUsername/NeuralNetwork/main/NeuralNetwork.lua" -- replace if you want
local AUTO_TRY_LOAD_EXTERNAL = false -- set true if you want to attempt loading external code first

-- Services
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local CoreGui = game:GetService("CoreGui")

-- Utility
local function shallowCopy(t)
    local r = {}
    for k,v in pairs(t) do r[k]=v end
    return r
end

local function deepCopy(orig)
    local lookup = {}
    local function _copy(obj)
        if type(obj) ~= "table" then return obj end
        if lookup[obj] then return lookup[obj] end
        local new = {}
        lookup[obj] = new
        for k,v in pairs(obj) do new[_copy(k)] = _copy(v) end
        return setmetatable(new, getmetatable(obj))
    end
    return _copy(orig)
end

local function isArray(t)
    if type(t) ~= "table" then return false end
    local n = 0
    for k,v in pairs(t) do
        if type(k) ~= "number" then return false end
        n = n + 1
    end
    return true
end

-- Optional external load (safe)
if AUTO_TRY_LOAD_EXTERNAL then
    local ok, res = pcall(function()
        local raw = game:HttpGet(GITHUB_RAW_URL, true)
        local f, err = loadstring(raw)
        if not f then error("loadstring error: "..tostring(err)) end
        return f()
    end)
    if ok then
        -- assume external script did everything and returned true/void
        print("[NeuralNet] External implementation loaded successfully.")
        -- Do not continue to redefine if external intentionally returned/handled everything.
        -- But for safety, we continue to define a local implementation below unless the external script set a global marker.
    else
        warn("[NeuralNet] External load failed, using local implementation. Error:", res)
    end
end

-- Neural Network Implementation
local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

-- Activation functions and derivatives
local Activations = {}

function Activations.sigmoid(x)
    return 1 / (1 + math.exp(-x))
end
function Activations.sigmoidDerivativeFromZ(z)
    local s = Activations.sigmoid(z)
    return s * (1 - s)
end

function Activations.relu(x)
    return math.max(0, x)
end
function Activations.reluDerivativeFromZ(z)
    return z > 0 and 1 or 0
end

function Activations.tanh(x)
    return math.tanh(x)
end
function Activations.tanhDerivativeFromZ(z)
    local t = math.tanh(z)
    return 1 - t * t
end

-- Helper to get activation and derivative functions by name
local function getActivationByName(name)
    name = (name or "sigmoid"):lower()
    if name == "sigmoid" then
        return Activations.sigmoid, Activations.sigmoidDerivativeFromZ
    elseif name == "relu" then
        return Activations.relu, Activations.reluDerivativeFromZ
    elseif name == "tanh" then
        return Activations.tanh, Activations.tanhDerivativeFromZ
    else
        return Activations.sigmoid, Activations.sigmoidDerivativeFromZ
    end
end

-- Constructor:
-- inputSize (number), hiddenLayers (array of numbers), outputSize (number), options table
-- options.learningRate, options.activation (string), options.initRange (number)
function NeuralNetwork.new(inputSize, hiddenLayers, outputSize, options)
    assert(type(inputSize) == "number" and inputSize >= 1, "inputSize must be number >=1")
    options = options or {}
    local self = setmetatable({}, NeuralNetwork)

    self.learningRate = options.learningRate or 0.1
    self.activationName = options.activation or "sigmoid"
    self.activation, self.activationDerivative = getActivationByName(self.activationName)
    self.initRange = options.initRange or 0.1 -- weights initialized in [-initRange, initRange]

    -- layerSizes is array: {inputSize, hidden1, hidden2, ..., outputSize}
    self.layerSizes = { inputSize }
    if type(hiddenLayers) == "table" then
        for _, s in ipairs(hiddenLayers) do
            assert(type(s) == "number" and s >= 1, "hidden layer sizes must be numbers >=1")
            table.insert(self.layerSizes, s)
        end
    end
    table.insert(self.layerSizes, outputSize)

    -- weights[l][j][i] => weight for layer l (from layer l to l+1), neuron j in next layer, input i from current layer
    self.weights = {}
    self.biases = {}
    for l = 1, #self.layerSizes - 1 do
        local inSize = self.layerSizes[l]
        local outSize = self.layerSizes[l+1]
        self.weights[l] = {}
        self.biases[l] = {}
        for j = 1, outSize do
            self.weights[l][j] = {}
            -- initialize weights
            for i = 1, inSize do
                -- uniform random in [-initRange, initRange]
                self.weights[l][j][i] = (math.random() * 2 - 1) * self.initRange
            end
            -- bias
            self.biases[l][j] = (math.random() * 2 - 1) * self.initRange
        end
    end

    return self
end

-- Forward pass: returns activations of all layers (array of arrays), and z values per layer (pre-activation sums)
function NeuralNetwork:forwardAll(input)
    assert(type(input) == "table", "input must be table")
    local activations = {}
    activations[1] = deepCopy(input)

    local zvals = {}

    for l = 1, #self.layerSizes - 1 do
        local inActs = activations[l]
        local outActs = {}
        local zlayer = {}
        for j = 1, self.layerSizes[l+1] do
            local sum = self.biases[l][j] or 0
            for i = 1, self.layerSizes[l] do
                sum = sum + (inActs[i] or 0) * (self.weights[l][j][i] or 0)
            end
            zlayer[j] = sum
            outActs[j] = self.activation(sum)
        end
        zvals[l] = zlayer
        activations[l+1] = outActs
    end

    return activations, zvals
end

-- Simple predict: returns output layer activation (array)
function NeuralNetwork:predict(input)
    local activations = self:forwardAll(input)
    return activations[1][#activations[1]] and activations[1][#activations[1]] or (self:forwardAll(input))
end

-- Backpropagation for a single sample (input, target)
-- input: array, target: array
function NeuralNetwork:backwardSingle(input, target)
    -- forward
    local activations, zvals = self:forwardAll(input)
    local L = #self.layerSizes - 1 -- number of weight layers
    -- delta[l][j] corresponds to layer l (1..L), neuron j in that next layer
    local delta = {}
    -- output layer delta
    delta[L] = {}
    for j = 1, self.layerSizes[#self.layerSizes] do
        local a = activations[#activations][j] or 0
        local err = a - (target[j] or 0)
        local deriv = self.activationDerivative(zvals[L][j] or 0)
        delta[L][j] = err * deriv
    end

    -- backpropagate
    for l = L-1, 1, -1 do
        delta[l] = {}
        for j = 1, self.layerSizes[l+1] do
            local err = 0
            for k = 1, self.layerSizes[l+2] do
                err = err + (delta[l+1][k] or 0) * (self.weights[l+1][k][j] or 0)
            end
            local deriv = self.activationDerivative(zvals[l][j] or 0)
            delta[l][j] = err * deriv
        end
    end

    -- update weights and biases (gradient descent)
    for l = 1, L do
        for j = 1, self.layerSizes[l+1] do
            for i = 1, self.layerSizes[l] do
                local grad = (delta[l][j] or 0) * (activations[l][i] or 0)
                self.weights[l][j][i] = (self.weights[l][j][i] or 0) - self.learningRate * grad
            end
            self.biases[l][j] = (self.biases[l][j] or 0) - self.learningRate * (delta[l][j] or 0)
        end
    end
end

-- Train: inputs: array of input arrays, targets: array of target arrays
-- options: epochs (number), shuffle (bool), verboseEvery (number of epochs for logging), batchSize (number or nil for full sample)
function NeuralNetwork:train(inputs, targets, options)
    options = options or {}
    local epochs = options.epochs or 1
    local shuffle = options.shuffle == nil and true or options.shuffle
    local verboseEvery = options.verboseEvery or math.max(1, math.floor(epochs / 10))
    local batchSize = options.batchSize or 1 -- 1 = stochastic gradient descent

    assert(#inputs == #targets, "inputs and targets must have same length")
    local n = #inputs
    for epoch = 1, epochs do
        -- create indices
        local indices = {}
        for i=1,n do indices[i] = i end
        if shuffle then
            for i = n, 2, -1 do
                local j = math.random(1, i)
                indices[i], indices[j] = indices[j], indices[i]
            end
        end

        -- mini-batch loop
        local idx = 1
        while idx <= n do
            local endIdx = math.min(n, idx + batchSize - 1)
            -- accumulate gradients by applying updates per sample (simple approach: apply per sample)
            for k = idx, endIdx do
                local sampleIndex = indices[k]
                self:backwardSingle(inputs[sampleIndex], targets[sampleIndex])
            end
            idx = endIdx + 1
        end

        if verboseEvery > 0 and epoch % verboseEvery == 0 then
            print(string.format("[NeuralNet] Epoch %d/%d complete", epoch, epochs))
        end
    end
end

-- Save network to JSON string (weights + biases + meta)
function NeuralNetwork:serialize()
    local data = {
        meta = {
            layerSizes = self.layerSizes,
            learningRate = self.learningRate,
            activationName = self.activationName,
            initRange = self.initRange
        },
        weights = self.weights,
        biases = self.biases
    }
    local ok, res = pcall(function() return HttpService:JSONEncode(data) end)
    if not ok then
        error("JSON serialization failed: "..tostring(res))
    end
    return res
end

-- Load network from JSON string. Returns NeuralNetwork instance.
function NeuralNetwork.deserialize(jsonString)
    assert(type(jsonString) == "string", "jsonString must be string")
    local ok, decoded = pcall(function() return HttpService:JSONDecode(jsonString) end)
    if not ok then error("JSON decode failed: "..tostring(decoded)) end
    local meta = decoded.meta
    assert(meta and meta.layerSizes, "missing meta.layerSizes in serialized data")

    local inputSize = meta.layerSizes[1]
    local hidden = {}
    for i = 2, #meta.layerSizes-1 do table.insert(hidden, meta.layerSizes[i]) end
    local outputSize = meta.layerSizes[#meta.layerSizes]

    local nn = NeuralNetwork.new(inputSize, hidden, outputSize, {
        learningRate = meta.learningRate or 0.1,
        activation = meta.activationName or "sigmoid",
        initRange = meta.initRange or 0.1
    })

    -- assign weights & biases directly (but keep shape checks)
    local function shapeMatches(decodedWeights, nnWeights)
        if type(decodedWeights) ~= "table" then return false end
        for l = 1, #nnWeights do
            if type(decodedWeights[tostring(l)]) == "table" then
                -- sometimes JSONDecode transforms numeric keys to strings; handle both
                decodedWeights = decodedWeights
            end
        end
        return true
    end

    -- If weights/biases were serialized as arrays, assign directly.
    nn.weights = decoded.weights or nn.weights
    nn.biases = decoded.biases or nn.biases

    -- Ensure numeric indices (convert keys that are strings to numeric where necessary)
    -- helper to normalize
    local function normalizeNestedArray(t)
        if type(t) ~= "table" then return t end
        -- if keys are string numbers, convert
        local keysAreStrings = false
        for k,v in pairs(t) do
            if type(k) == "string" then
                keysAreStrings = true
                break
            end
        end
        if keysAreStrings then
            local new = {}
            for k,v in pairs(t) do
                local nk = tonumber(k) or k
                new[nk] = normalizeNestedArray(v)
            end
            return new
        end
        -- recursively normalize children
        for k,v in pairs(t) do t[k] = normalizeNestedArray(v) end
        return t
    end

    nn.weights = normalizeNestedArray(nn.weights)
    nn.biases = normalizeNestedArray(nn.biases)

    return nn
end

-- Convenience: save to DataStore (optional) - user can adapt
-- NOTE: DataStore usage is commented out because it requires proper handling and limits.
--[[
local DataStoreService = game:GetService("DataStoreService")
function NeuralNetwork:saveToDataStore(key, scopeName)
    local ds = DataStoreService:GetDataStore(scopeName or "NeuralNetStore")
    local s = self:serialize()
    ds:SetAsync(key, s)
end
function NeuralNetwork.loadFromDataStore(key, scopeName)
    local ds = DataStoreService:GetDataStore(scopeName or "NeuralNetStore")
    local s = ds:GetAsync(key)
    if s then
        return NeuralNetwork.deserialize(s)
    end
    return nil
end
--]]

-- ========== Logger + GUI + Example usage ==========

-- Global control state
local LOG_ENABLED = true
local LOG_INTERVAL = 10 -- seconds

-- Build GUI (safe: attempt CoreGui; if denied, fallback to PlayerGui for local player)
local function createGUI()
    local parent = nil
    -- try CoreGui first (may be restricted). If cannot set, fallback.
    local succeed, err = pcall(function() parent = CoreGui end)
    if not succeed or not parent then
        local localPlayer = Players.LocalPlayer
        if localPlayer and localPlayer:FindFirstChild("PlayerGui") then
            parent = localPlayer:FindFirstChild("PlayerGui")
        else
            parent = nil
        end
    end

    if not parent then
        warn("[NeuralNet] Unable to place GUI (no CoreGui or PlayerGui available).")
        return nil
    end

    -- Create screen gui
    local screenGui = Instance.new("ScreenGui")
    screenGui.Name = "NeuralNetGUI_" .. tostring(tick())
    screenGui.ResetOnSpawn = false
    screenGui.Parent = parent

    local frame = Instance.new("Frame")
    frame.Size = UDim2.new(0, 260, 0, 120)
    frame.Position = UDim2.new(0.02, 0, 0.02, 0)
    frame.BackgroundColor3 = Color3.fromRGB(28,28,28)
    frame.BorderSizePixel = 0
    frame.Parent = screenGui
    frame.Active = true
    frame.Draggable = true

    local title = Instance.new("TextLabel")
    title.Size = UDim2.new(1, -12, 0, 28)
    title.Position = UDim2.new(0, 6, 0, 6)
    title.BackgroundTransparency = 1
    title.Text = "NeuralNet (Logger)"
    title.Font = Enum.Font.SourceSansSemibold
    title.TextSize = 18
    title.TextColor3 = Color3.fromRGB(240,240,240)
    title.Parent = frame

    local toggle = Instance.new("TextButton")
    toggle.Size = UDim2.new(0, 120, 0, 36)
    toggle.Position = UDim2.new(0, 6, 0, 40)
    toggle.Text = LOG_ENABLED and "Logging: ON" or "Logging: OFF"
    toggle.Font = Enum.Font.SourceSans
    toggle.TextSize = 14
    toggle.Parent = frame

    local intervalLabel = Instance.new("TextLabel")
    intervalLabel.Size = UDim2.new(0, 120, 0, 20)
    intervalLabel.Position = UDim2.new(0, 140, 0, 44)
    intervalLabel.BackgroundTransparency = 1
    intervalLabel.Text = "Interval: " .. tostring(LOG_INTERVAL) .. "s"
    intervalLabel.Font = Enum.Font.SourceSans
    intervalLabel.TextSize = 13
    intervalLabel.TextColor3 = Color3.fromRGB(220,220,220)
    intervalLabel.Parent = frame

    local inc = Instance.new("TextButton")
    inc.Size = UDim2.new(0, 36, 0, 28)
    inc.Position = UDim2.new(0, 140, 0, 68)
    inc.Text = "+"
    inc.Font = Enum.Font.SourceSans
    inc.TextSize = 18
    inc.Parent = frame

    local dec = Instance.new("TextButton")
    dec.Size = UDim2.new(0, 36, 0, 28)
    dec.Position = UDim2.new(0, 182, 0, 68)
    dec.Text = "-"
    dec.Font = Enum.Font.SourceSans
    dec.TextSize = 18
    dec.Parent = frame

    local saveBtn = Instance.new("TextButton")
    saveBtn.Size = UDim2.new(0, 120, 0, 28)
    saveBtn.Position = UDim2.new(0, 6, 0, 76)
    saveBtn.Text = "Export JSON"
    saveBtn.Font = Enum.Font.SourceSans
    saveBtn.TextSize = 14
    saveBtn.Parent = frame

    -- Button connections
    toggle.MouseButton1Click:Connect(function()
        LOG_ENABLED = not LOG_ENABLED
        toggle.Text = LOG_ENABLED and "Logging: ON" or "Logging: OFF"
    end)
    inc.MouseButton1Click:Connect(function()
        LOG_INTERVAL = math.max(1, LOG_INTERVAL - 1) -- plus/minus reversed to reflect UI (+ reduces interval)
        intervalLabel.Text = "Interval: " .. tostring(LOG_INTERVAL) .. "s"
    end)
    dec.MouseButton1Click:Connect(function()
        LOG_INTERVAL = LOG_INTERVAL + 1
        intervalLabel.Text = "Interval: " .. tostring(LOG_INTERVAL) .. "s"
    end)
    saveBtn.MouseButton1Click:Connect(function()
        -- produce JSON and copy to clipboard if possible; otherwise print length and first/last chars
        local ok, json = pcall(function() return NeuralNetworkInstance:serialize() end)
        if ok and json then
            -- Try to copy to clipboard via SetClipboard (available in Studio)
            local success, errmsg = pcall(function() setclipboard(json) end)
            if success then
                print("[NeuralNet GUI] JSON copied to clipboard.")
            else
                -- fallback: print truncated
                print("[NeuralNet GUI] JSON length:", #json, " (first 200 chars):")
                print(string.sub(json,1,200))
            end
        else
            warn("[NeuralNet GUI] Failed to serialize network:", json)
        end
    end)

    return screenGui
end

-- Create a default network instance (example sizes). You can replace sizes when constructing from your code.
local NeuralNetworkInstance = NeuralNetwork.new(2, {4}, 1, {
    learningRate = 0.1,
    activation = "sigmoid",
    initRange = 0.1
})

-- Logger loop
task.spawn(function()
    while true do
        local interval = LOG_INTERVAL or 10
        task.wait(interval)
        if LOG_ENABLED then
            -- Provide some useful info about network
            local info = string.format("[NeuralNet] active — layers: %d (sizes: %s) lr=%.4f",
                #NeuralNetworkInstance.layerSizes,
                table.concat(NeuralNetworkInstance.layerSizes, ","),
                NeuralNetworkInstance.learningRate
            )
            print(info)
        end
    end
end)

-- Optional: apply predicted output to local player's humanoid (safe-guarded)
local function tryApplyToLocalHumanoid()
    local player = Players.LocalPlayer
    if not player then return end
    local character = player.Character or player.CharacterAdded:Wait()
    if not character then return end
    local hrp = character:FindFirstChild("HumanoidRootPart")
    local humanoid = character:FindFirstChildOfClass("Humanoid")
    if not hrp or not humanoid then return end

    RunService.RenderStepped:Connect(function()
        if LOG_ENABLED then
            local input = { hrp.Position.X, hrp.Position.Y }
            local out = NeuralNetworkInstance:predict(input)
            if type(out) == "table" and out[1] then
                -- clamp and set WalkSpeed safely
                local value = tonumber(out[1]) or 0
                local ws = 16 + (value * 10)
                if humanoid and humanoid.Parent then
                    humanoid.WalkSpeed = math.clamp(ws, 6, 100)
                end
            end
        end
    end)
end

-- Build GUI (non-blocking)
task.spawn(function()
    local gui = createGUI()
    if gui then
        print("[NeuralNet] GUI created.")
    end
end)

-- Example: train small XOR dataset in background (non-blocking)
task.spawn(function()
    local inputs = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    }
    local targets = {
        {0},
        {1},
        {1},
        {0}
    }
    print("[NeuralNet] Starting sample training (XOR) — this runs in background.")
    NeuralNetworkInstance:train(inputs, targets, { epochs = 500, shuffle = true, verboseEvery = 100, batchSize = 1 })
    print("[NeuralNet] Sample training complete.")
end)

-- Public API table (so users can access from other scripts if they require this module)
local API = {
    NeuralNetwork = NeuralNetwork,
    new = function(...) return NeuralNetwork.new(...) end,
    deserialize = function(json) return NeuralNetwork.deserialize(json) end,
    instance = NeuralNetworkInstance,
    setLogInterval = function(sec) LOG_INTERVAL = math.max(1, tonumber(sec) or LOG_INTERVAL) end,
    enableLog = function(b) LOG_ENABLED = (b and true) end,
    disableLog = function() LOG_ENABLED = false end
}

-- Expose to _G for easy interactive usage (optional)
_G.NeuralNetAPI = API

print("[NeuralNet] Ready. API available as _G.NeuralNetAPI. Instance:", NeuralNetworkInstance)

-- Return API for require-style usage when loaded as ModuleScript (if used that way)
return API
