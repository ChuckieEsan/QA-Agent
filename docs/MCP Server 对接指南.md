# MCP Server 对接指南

本文档为第三方开发者提供 GovMcpGateway MCP Server 的对接说明，包含连接方式、协议流程和工具定义。

---

## 1. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI Agent / Client                              │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ MCP Protocol (HTTP/SSE)
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                      GovMcpGateway (Port: 8083)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  MCP Server Endpoints                                              ││
│  │  - /sse        : 建立 SSE 连接，获取 sessionId                     ││
│  │  - /mcp/message: 发送 JSON-RPC 请求                                ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  Tool Execution                                                    ││
│  │  - DubboGenericStrategy: 调用后端 Dubbo 服务                       ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ Dubbo Protocol
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│              GovServiceMock (Port: 8082, Dubbo: 20880)                  │
│  - WorkOrderService: 工单管理服务                                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      Nacos (Port: 8848)                                 │
│  - 配置中心: gov-service-mock-mcp.yml                                   │
│  - 服务发现: GovServiceMock 注册                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 快速开始

### 2.1 启动服务

**前置条件：**
- JDK 21+
- Nacos 运行在 localhost:8848

**启动步骤：**

```bash
# 1. 启动 Nacos（如果未启动）
# 参考: https://nacos.io/docs/latest/quick-start/quick-start/

# 2. 启动 GovServiceMock（提供后端业务服务）
cd GovServiceMock
mvn spring-boot:run

# 3. 启动 GovMcpGateway（MCP Server）
cd GovMcpGateway
mvn spring-boot:run
```

**服务端口：**
| 服务 | 端口 | 说明 |
|-----|------|-----|
| GovMcpGateway | 8083 | MCP Server 端点 |
| GovServiceMock | 8082 | 后端业务服务（REST API） |
| GovServiceMock | 20880 | Dubbo 协议端口 |
| Nacos | 8848 | 配置中心 & 服务发现 |

### 2.2 验证服务

```bash
# 健康检查
curl http://localhost:8083/health
# 返回: {"status":"UP"}

# SSE 端点（建立连接获取 sessionId）
curl http://localhost:8083/sse
# 返回类似: event:endpoint\ndata:/mcp/message?sessionId=xxx
```

---

## 3. MCP 协议对接

### 3.1 连接流程

MCP 协议采用 **SSE (Server-Sent Events)** 长连接方式通信：

```
1. Client ──HTTP GET──▶ Server   (/sse)
2. Client ◀──SSE─── Server        (返回 sessionId)

3. Client ──HTTP POST──▶ Server   (/mcp/message?sessionId=xxx)
                                        │
4. Client ◀──SSE─── Server        (推送 JSON-RPC 响应)
```

### 3.2 初始化握手（必须）

**重要**：在发送工具调用前，必须先完成 MCP 协议握手。

**Step 1: 发送 initialize 请求**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "your-client-name",
      "version": "1.0.0"
    }
  }
}
```

**Step 2: 等待服务端响应**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {}
    },
    "serverInfo": {
      "name": "spring-ai-mcp-server",
      "version": "1.1.2"
    }
  }
}
```

**Step 3: 发送 notifications/initialized 通知**

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized",
  "params": {}
}
```

### 3.3 工具列表查询

初始化完成后，可以查询可用工具：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

### 3.4 工具调用

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "query_gov_work_order",
    "arguments": {
      "orderId": "GD_2024_001"
    }
  }
}
```

---

## 4. 工具定义 (Schema)

### 4.1 工具总览

| 工具名称 | 功能 | 调用方式 |
|---------|------|---------|
| `create_gov_work_order` | 创建政务工单 | Dubbo |
| `query_gov_work_order` | 查询工单详情 | Dubbo |
| `process_gov_work_order` | 工单流转处理 | Dubbo |
| `submit_work_order_feedback` | 提交用户评价 | Dubbo |

---

### 4.2 create_gov_work_order - 创建工单

**功能描述**：用户提交诉求后，AI Agent 调用此接口创建工单。

**参数 Schema**：

```json
{
  "type": "object",
  "properties": {
    "orderData": {
      "type": "object",
      "description": "工单详细数据",
      "properties": {
        "userId": {
          "type": "string",
          "description": "提交人ID"
        },
        "userPhone": {
          "type": "string",
          "description": "联系方式"
        },
        "title": {
          "type": "string",
          "description": "工单标题"
        },
        "content": {
          "type": "string",
          "description": "详细描述"
        },
        "department": {
          "type": "string",
          "description": "责任部门"
        },
        "elements": {
          "type": "object",
          "description": "五大核心要素",
          "properties": {
            "time": {
              "type": "string",
              "description": "时间"
            },
            "location": {
              "type": "string",
              "description": "地点"
            },
            "event": {
              "type": "string",
              "description": "核心事件"
            },
            "goal": {
              "type": "string",
              "description": "诉求目标"
            },
            "subjects": {
              "type": "array",
              "description": "涉及主体",
              "items": {
                "type": "string"
              }
            }
          }
        }
      },
      "required": ["userId", "userPhone", "title", "content"]
    }
  },
  "required": ["orderData"]
}
```

**调用示例**：

```json
{
  "name": "create_gov_work_order",
  "arguments": {
    "orderData": {
      "userId": "user001",
      "userPhone": "13800138000",
      "title": "关于路灯损坏的投诉",
      "content": "XX路XX号路灯损坏，存在安全隐患",
      "department": "城管局",
      "elements": {
        "time": "2024-01-15",
        "location": "XX路XX号",
        "event": "路灯损坏",
        "goal": "尽快修复",
        "subjects": ["城管局", "供电局"]
      }
    }
  }
}
```

**返回格式**：

```json
{
  "success": true,
  "message": "工单创建成功",
  "data": {
    "orderId": "GD_2024_XXXXXX",
    "status": "UNASSIGNED",
    "createTime": "2024-01-15T10:30:00"
  }
}
```

---

### 4.3 query_gov_work_order - 查询工单

**功能描述**：根据工单ID查询政务工单详情，返回工单的基本信息、当前状态、处理进度等。

**参数 Schema**：

```json
{
  "type": "object",
  "properties": {
    "orderId": {
      "type": "string",
      "description": "工单唯一编号"
    }
  },
  "required": ["orderId"]
}
```

**调用示例**：

```json
{
  "name": "query_gov_work_order",
  "arguments": {
    "orderId": "GD_2024_001"
  }
}
```

**返回格式**：

```json
{
  "success": true,
  "message": "查询成功",
  "data": {
    "id": "GD_2024_001",
    "title": "关于路灯损坏的投诉",
    "content": "XX路XX号路灯损坏，存在安全隐患",
    "status": "PROCESSING",
    "department": "城管局",
    "handler": "张三",
    "createTime": "2024-01-15T10:30:00",
    "updateTime": "2024-01-15T14:20:00"
  }
}
```

---

### 4.4 process_gov_work_order - 工单流转处理

**功能描述**：工单流转处理，包括分拨、受理工单、回复用户等操作。

**参数 Schema**：

```json
{
  "type": "object",
  "properties": {
    "orderId": {
      "type": "string",
      "description": "工单唯一编号"
    },
    "action": {
      "type": "string",
      "description": "操作类型",
      "enum": ["ASSIGN", "ACCEPT", "REPLY"]
    },
    "payload": {
      "type": "object",
      "description": "操作数据",
      "properties": {
        "department": {
          "type": "string",
          "description": "分拨部门(ASSIGN时必填)"
        },
        "handler": {
          "type": "string",
          "description": "经办人"
        },
        "replyContent": {
          "type": "string",
          "description": "回复内容(REPLY时必填)"
        }
      }
    }
  },
  "required": ["orderId", "action", "payload"]
}
```

**调用示例**：

```json
{
  "name": "process_gov_work_order",
  "arguments": {
    "orderId": "GD_2024_001",
    "action": "ASSIGN",
    "payload": {
      "department": "环保局",
      "handler": "李四"
    }
  }
}
```

```json
{
  "name": "process_gov_work_order",
  "arguments": {
    "orderId": "GD_2024_001",
    "action": "ACCEPT",
    "payload": {
      "handler": "李四"
    }
  }
}
```

```json
{
  "name": "process_gov_work_order",
  "arguments": {
    "orderId": "GD_2024_001",
    "action": "REPLY",
    "payload": {
      "replyContent": "已安排人员处理，预计3个工作日内完成"
    }
  }
}
```

**返回格式**：

```json
{
  "success": true,
  "message": "工单已分拨",
  "data": {
    "orderId": "GD_2024_001",
    "status": "ASSIGNED",
    "department": "环保局",
    "handler": "李四"
  }
}
```

---

### 4.5 submit_work_order_feedback - 提交评价

**功能描述**：工单处理完成后，用户可以对处理结果进行评价。

**参数 Schema**：

```json
{
  "type": "object",
  "properties": {
    "orderId": {
      "type": "string",
      "description": "工单唯一编号"
    },
    "feedbackData": {
      "type": "object",
      "description": "评价数据",
      "properties": {
        "timeliness": {
          "type": "integer",
          "description": "受理时效评分(1-5分)",
          "minimum": 1,
          "maximum": 5
        },
        "attitude": {
          "type": "integer",
          "description": "办理态度评分(1-5分)",
          "minimum": 1,
          "maximum": 5
        },
        "result": {
          "type": "integer",
          "description": "处理结果评分(1-5分)",
          "minimum": 1,
          "maximum": 5
        },
        "comment": {
          "type": "string",
          "description": "文字评价"
        }
      },
      "required": ["timeliness", "attitude", "result"]
    }
  },
  "required": ["orderId", "feedbackData"]
}
```

**调用示例**：

```json
{
  "name": "submit_work_order_feedback",
  "arguments": {
    "orderId": "GD_2024_001",
    "feedbackData": {
      "timeliness": 5,
      "attitude": 4,
      "result": 5,
      "comment": "处理及时，非常满意！"
    }
  }
}
```

**返回格式**：

```json
{
  "success": true,
  "message": "评价提交成功",
  "data": {
    "orderId": "GD_2024_001",
    "feedbackTime": "2024-01-15T16:30:00"
  }
}
```

---

## 5. 错误响应

所有工具调用可能返回以下错误格式：

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": "详细错误信息"
  }
}
```

**错误码说明**：

| 错误码 | 含义 |
|-------|------|
| -32600 | 无效的请求 |
| -32601 | 方法未找到 |
| -32602 | 无效的参数 |
| -32603 | 内部错误 |

---

## 6. 示例代码

### 6.1 Python 示例

```python
import requests
import sseclient
import json

class McpClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session_id = None

    def connect(self):
        """建立 SSE 连接"""
        response = requests.get(f"{self.base_url}/sse", stream=True)
        client = sseclient.SSEClient(response)

        for event in client.events():
            if event.event == "endpoint":
                data = event.data
                self.session_id = data.split("sessionId=")[1]
                print(f"Connected, sessionId: {self.session_id}")
                break

    def initialize(self):
        """MCP 初始化握手"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "python-client", "version": "1.0.0"}
            }
        }
        response = self._send_request(request)
        print("Initialized:", response)

        # 发送 notifications/initialized
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        self._send_request(notification)

    def list_tools(self):
        """查询工具列表"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        return self._send_request(request)

    def call_tool(self, name: str, arguments: dict):
        """调用工具"""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        return self._send_request(request)

    def _send_request(self, request: dict):
        """发送请求"""
        url = f"{self.base_url}/mcp/message?sessionId={self.session_id}"
        response = requests.post(url, json=request)
        return response.json()

# 使用示例
client = McpClient("http://localhost:8083")
client.connect()
client.initialize()

# 查询工具列表
tools = client.list_tools()
print("Available tools:", tools)

# 调用工具
result = client.call_tool("query_gov_work_order", {"orderId": "GD_2024_001"})
print("Query result:", result)
```

### 6.2 JavaScript 示例

```javascript
class McpClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.sessionId = null;
    this.eventSource = null;
    this.responses = new Map();
  }

  async connect() {
    return new Promise((resolve) => {
      this.eventSource = new EventSource(`${this.baseUrl}/sse`);

      this.eventSource.onmessage = (event) => {
        const data = event.data;
        if (data.includes('sessionId=')) {
          this.sessionId = data.split('sessionId=')[1];
          console.log('Connected, sessionId:', this.sessionId);
          resolve();
        }
      };

      this.eventSource.onerror = (err) => {
        console.error('SSE error:', err);
      };
    });
  }

  async sendRequest(request) {
    const response = await fetch(
      `${this.baseUrl}/mcp/message?sessionId=${this.sessionId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      }
    );
    return response.json();
  }

  async initialize() {
    const request = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'js-client', version: '1.0.0' }
      }
    };
    const result = await this.sendRequest(request);
    console.log('Initialized:', result);

    // 发送 notifications/initialized
    await this.sendRequest({
      jsonrpc: '2.0',
      method: 'notifications/initialized',
      params: {}
    });
  }

  async listTools() {
    return this.sendRequest({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/list',
      params: {}
    });
  }

  async callTool(name, arguments_) {
    return this.sendRequest({
      jsonrpc: '2.0',
      id: 3,
      method: 'tools/call',
      params: { name, arguments: arguments_ }
    });
  }
}

// 使用示例
const client = new McpClient('http://localhost:8083');
await client.connect();
await client.initialize();

const tools = await client.listTools();
console.log('Available tools:', tools);

const result = await client.callTool('query_gov_work_order', {
  orderId: 'GD_2024_001'
});
console.log('Query result:', result);
```

---

## 7. 测试数据

系统预置了以下测试工单：

| 工单ID | 标题 | 状态 |
|--------|------|------|
| GD_2024_001 | 关于路灯损坏的投诉 | PROCESSING |
| GD_2024_002 | 噪音扰民投诉 | COMPLETED |

---

## 8. 常见问题

### Q1: 为什么请求总是超时？

**原因**：MCP 协议需要先完成初始化握手才能调用工具。

**解决**：确保按照 3.2 节的步骤完成 `initialize` 请求和 `notifications/initialized` 通知。

### Q2: 如何查看服务日志？

```bash
# Gateway 日志
tail -f GovMcpGateway/logs/xxx.log

# 服务端 SSE 响应
curl -v http://localhost:8083/sse
```

### Q3: 如何确认 Dubbo 服务可用？

```bash
# 查看 Nacos 注册的服务
curl http://localhost:8848/nacos/v1/ns/instance/list?serviceName=gov-service-mock
```

---

## 9. 相关配置

### Nacos 配置 (gov-service-mock-mcp.yml)

```yaml
agent:
  tools:
    - name: "create_gov_work_order"
      type: "DUBBO"
      # ... 其他配置
```

### Gateway 端口配置

```yaml
server:
  port: 8083

spring:
  ai:
    mcp:
      server:
        sse:
          path: /sse
        message:
          path: /mcp/message
```

---

## 10. 技术支持

如有问题，请检查：
1. Nacos 是否正常运行 (http://localhost:8848)
2. GovServiceMock 是否已启动 (端口 8082)
3. GovMcpGateway 是否已启动 (端口 8083)
4. 服务是否正确注册到 Nacos